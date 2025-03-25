import os
import sys
import logging
import json
from datetime import datetime
from typing import Dict, List, Optional
import traceback

from sqlalchemy import create_engine, MetaData, inspect
from sqlalchemy.exc import SQLAlchemyError
from openai import OpenAI
from dotenv import load_dotenv
from tqdm import tqdm

class ForeignKeyAnalyzerError(Exception):
    """Base exception class for ForeignKeyAnalyzer"""
    pass

class ConfigurationError(ForeignKeyAnalyzerError):
    """Raised when there are configuration issues"""
    pass

def load_config() -> Dict:
    """Load configuration from .env file"""
    load_dotenv()
    
    config = {}
    
    # Required variables
    required_vars = [
        'DB_CONNECTION_STRING',
        'OPENAI_API_KEY',
        'AI_MODEL'
    ]

    # Check for required variables
    missing_vars = []
    for var in required_vars:
        value = os.getenv(var)
        if value is None:
            missing_vars.append(var)
        config[var] = value

    if missing_vars:
        raise ConfigurationError(f"Missing required environment variables: {', '.join(missing_vars)}")

    # Optional variables with defaults
    config['LOG_LEVEL'] = os.getenv('LOG_LEVEL', 'DEBUG')
    config['BATCH_SIZE'] = int(os.getenv('BATCH_SIZE', '10'))
    config['CONFIDENCE_THRESHOLD'] = float(os.getenv('CONFIDENCE_THRESHOLD', '0.8'))
    config['MAX_RETRIES'] = int(os.getenv('MAX_RETRIES', '3'))
    
    # Parse ignored columns list
    ignored_columns_str = os.getenv('IGNORED_COLUMNS', '')
    config['IGNORED_COLUMNS'] = [col.strip() for col in ignored_columns_str.split(',') if col.strip()]

    config['DB_INSTRUCTIONS'] = os.getenv('DB_INSTRUCTIONS', '')
    config['DB_DESCRIPTION'] = os.getenv('DB_DESCRIPTION', '')
    return config

def setup_logging(log_level: str) -> logging.Logger:
    """Setup logging configuration"""
    # Create logs directory if it doesn't exist
    if not os.path.exists('logs'):
        os.makedirs('logs')

    # Create logger
    logger = logging.getLogger('foreign_key_analyzer')
    logger.setLevel(getattr(logging, log_level))
    logger.handlers = []  # Remove any existing handlers

    # File handler
    timestamp = datetime.now().strftime('%Y%m%d')
    file_handler = logging.FileHandler(f'logs/foreign_key_analysis_{timestamp}.log')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)

    # Console handler (errors only)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.ERROR)
    console_handler.setFormatter(logging.Formatter('ERROR: %(message)s'))
    logger.addHandler(console_handler)

    return logger

class ForeignKeyAnalyzer:
    def __init__(self, config: Dict, logger: logging.Logger):
        """Initialize the ForeignKeyAnalyzer"""
        self.config = config
        self.logger = logger
        self.engine = None
        self.connection = None
        self.inspector = None
        self.metadata = None
        self.openai_client = None
        self.total_tokens = 0
        self.sql_file = None
        self.suggestion_cache = {}  # Cache for storing previous suggestions
        self.rejected_cache = {}  # Cache for storing rejected suggestions
        self._setup_output_file()

    def _setup_output_file(self):
        """Create the output directory and SQL file for foreign key definitions"""
        os.makedirs('output', exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'output/foreign_key_definitions_{timestamp}.sql'
        self.sql_file = open(filename, 'w')
        self.logger.info(f"SQL output will be saved to: {filename}")

    def connect_to_db(self) -> bool:
        """Connect to the database using SQLAlchemy"""
        try:
            self.engine = create_engine(self.config['DB_CONNECTION_STRING'])
            self.connection = self.engine.connect()
            self.inspector = inspect(self.engine)
            self.metadata = MetaData()
            self.metadata.reflect(bind=self.engine)
            return True
        except SQLAlchemyError as e:
            error_msg = f"Error connecting to database: {str(e)}"
            self.logger.error(error_msg)
            raise ForeignKeyAnalyzerError(error_msg)

    def get_table_info(self, table_name: str) -> Dict:
        """Get detailed information about a table including columns and existing foreign keys"""
        columns = self.inspector.get_columns(table_name)
        foreign_keys = self.inspector.get_foreign_keys(table_name)
        primary_key = self.inspector.get_pk_constraint(table_name)
        
        return {
            'columns': columns,
            'foreign_keys': foreign_keys,
            'primary_key': primary_key
        }

    def find_potential_foreign_keys(self) -> List[Dict]:
        """Find columns that might be foreign keys (ending with _id) but don't have constraints"""
        potential_fks = []
        tables = self.inspector.get_table_names()
        
        for table_name in tables:
            table_info = self.get_table_info(table_name)
            existing_fk_columns = {col for fk in table_info['foreign_keys'] 
                                 for col in fk['constrained_columns']}
            
            for column in table_info['columns']:
                col_name = column['name']
                
                # Skip if column matches any ignore pattern
                if any(ignored_pattern in col_name.lower() 
                      for ignored_pattern in self.config['IGNORED_COLUMNS']):
                    self.logger.debug(f"Skipping ignored column {table_name}.{col_name}")
                    continue
                    
                if (col_name.lower().endswith('_id') and 
                    col_name not in existing_fk_columns):
                    potential_fks.append({
                        'table': table_name,
                        'column': col_name,
                        'type': str(column['type'])
                    })

        self.logger.info(f"Found {len(potential_fks)} potential foreign key columns")

        return potential_fks

    def _get_cache_key(self, column_name: str) -> str:
        """Generate a cache key from a column name by extracting the prefix before '_id'"""
        # Convert to lowercase for case-insensitive matching
        column_name = column_name.lower()
        if column_name.endswith('_id'):
            return column_name[:-3]  # Remove '_id' suffix
        return column_name

    def _get_rejection_cache_key(self, table_name: str, column_name: str) -> str:
        """Generate a cache key for rejected suggestions"""
        return f"{table_name}.{column_name}"

    def analyze_potential_relationship(self, table_name: str, column_name: str,
                                    all_tables: List[str]) -> Optional[Dict]:
        """Use OpenAI to analyze a potential foreign key relationship"""
        # Check if this relationship was previously rejected
        rejection_key = self._get_rejection_cache_key(table_name, column_name)
        if rejection_key in self.rejected_cache:
            self.logger.info(f"Skipping previously rejected suggestion for {rejection_key}")
            return None

        # Check cache first
        cache_key = self._get_cache_key(column_name)
        if cache_key in self.suggestion_cache:
            cached_suggestion = self.suggestion_cache[cache_key].copy()
            cached_suggestion['from_cache'] = True
            self.logger.info(f"Using cached suggestion for {table_name}.{column_name}")
            return cached_suggestion

        if not self.openai_client:
            self.openai_client = OpenAI()

        # Get existing foreign keys for context
        existing_fks = []
        for table in all_tables:
            fks = self.inspector.get_foreign_keys(table)
            for fk in fks:
                existing_fks.append({
                    'source_table': table,
                    'source_column': fk['constrained_columns'][0],
                    'target_table': fk['referred_table'],
                    'target_column': fk['referred_columns'][0]
                })

        retries = 0
        max_retries = self.config['MAX_RETRIES']

        while retries < max_retries:
            # Prepare the prompt
            prompt = f"""Analyze the potential foreign key relationship for the following column:

Source Table: {table_name}
Source Column: {column_name}


{self.config['DB_DESCRIPTION'] != '' and f"Database description: {self.config['DB_DESCRIPTION']}\n" or ''}
{self.config['DB_INSTRUCTIONS'] != '' and f"Additional instructions: {self.config['DB_INSTRUCTIONS']}\n" or ''}
Available tables in the database:
{json.dumps(all_tables, indent=2)}

Existing foreign key relationships in the database:
{json.dumps(existing_fks, indent=2)}

Based on the naming convention and existing relationships, suggest the most likely table and column that this foreign key should reference. Consider common patterns in the existing foreign keys and table relationships.

IMPORTANT: You MUST ONLY suggest tables from the provided list of available tables.

Confidence scoring rules (apply in order, use the FIRST matching rule):
1. If there is an existing foreign key in the database with EXACTLY the same source column name, set confidence to 1.0
2. If the column name EXACTLY matches the target table name (e.g. user_id -> user), set confidence to 1.0
3. If the column name is a PARTIAL match (e.g. user_id -> app_user OR app_user_id -> user), set confidence to 0.7
4. If the source table name is used for the reasoning (e.g. "company" in user_company.entity_id -> company.id), set confidence to 0.7
5. If the column name suggests a generic relationship (e.g. parent_id, reference_id) and you can determine the likely table, set confidence to 0.5
6. If ANY assumptions are made, set confidence to 0.5
7. If you cannot find a suitable table or are highly uncertain, set confidence to 0.0

Return a JSON object with the following structure:
{{
    "suggested_table": "table_name",
    "suggested_column": "column_name",
    "reasoning": "explanation of the suggestion",
    "confidence": 0.0-1.0
}}

DO NOT make up or suggest tables that don't exist in the database. DO NOT make up foreign keys that don't exist in the database. ONLY use foreign keys if they match the source column name.
Make a best effort guess from the existing foreign keys and list of tables.
USE the above rules to set the confidence score. REMEMBER if you are making ANY assumptions, the confidence score will be low - only EXACT matches will score 1.0.

IMPORTANT: Your response must be valid JSON. Do not include any additional text before or after the JSON object."""

            self.logger.debug(f"Sending prompt to OpenAI for {table_name}.{column_name} (attempt {retries + 1}/{max_retries}):")
            self.logger.debug("Prompt:")
            self.logger.debug(prompt)

            try:
                messages = [
                    {"role": "system", "content": "You are a database expert analyzing foreign key relationships. You must respond with ONLY a valid JSON object, no additional text."},
                    {"role": "user", "content": prompt}
                ]

                response = self.openai_client.chat.completions.create(
                    model=self.config['AI_MODEL'],
                    messages=messages,
                    temperature=0.1,
                    response_format={"type": "json_object"}  # Force JSON response
                )

                self.total_tokens += response.usage.total_tokens
                self.logger.debug(f"OpenAI API usage stats for {table_name}.{column_name}: {response.usage}")
                
                response_content = response.choices[0].message.content.strip()
                self.logger.debug(f"Raw response from OpenAI for {table_name}.{column_name}:")
                self.logger.debug(response_content)
                
                try:
                    suggestion = json.loads(response_content)
                    suggestion['from_cache'] = False

                    # Validate that the suggested table exists
                    if suggestion['suggested_table'] not in all_tables:
                        self.logger.warning(f"AI suggested non-existent table '{suggestion['suggested_table']}'. Retrying...")
                        retries += 1
                        continue

                    self.logger.debug(f"Parsed suggestion for {table_name}.{column_name}:")
                    self.logger.debug(json.dumps(suggestion, indent=2))
                    return suggestion

                except json.JSONDecodeError as e:
                    self.logger.error(f"Failed to parse JSON response for {table_name}.{column_name}: {e}")
                    self.logger.error(f"Response content that failed to parse: {response_content}")
                    retries += 1
                    continue

            except Exception as e:
                self.logger.error(f"Error in OpenAI API call for {table_name}.{column_name}: {e}")
                self.logger.error(traceback.format_exc())
                retries += 1
                continue

        # If we've exhausted all retries, return None
        self.logger.error(f"Failed to get valid suggestion for {table_name}.{column_name} after {max_retries} attempts")
        return None

    def generate_foreign_key_sql(self, source_table: str, source_column: str,
                               target_table: str, target_column: str) -> str:
        """Generate SQL to add a foreign key constraint"""
        constraint_name = f"fk_{source_table}_{source_column}"
        sql = f"""ALTER TABLE {source_table} ADD CONSTRAINT {constraint_name} FOREIGN KEY ({source_column}) REFERENCES {target_table} ({target_column});"""
        return sql

    def analyze_and_suggest(self):
        """Main method to analyze and suggest foreign key relationships"""
        try:
            # Get all tables
            tables = self.inspector.get_table_names()
            self.logger.info(f"Found {len(tables)} tables in the database")

            # Find potential foreign keys
            potential_fks = self.find_potential_foreign_keys()
            self.logger.info(f"Found {len(potential_fks)} potential foreign key columns")

            # Create progress bar
            with tqdm(total=len(potential_fks), desc="Analyzing foreign keys") as pbar:
                for potential_fk in potential_fks:
                    table_name = potential_fk['table']
                    column_name = potential_fk['column']

                    pbar.set_description(f"Analyzing {table_name}.{column_name}")
                    self.logger.info(f"Analyzing {table_name}.{column_name}")
                    
                    # Get suggestion from OpenAI
                    suggestion = self.analyze_potential_relationship(
                        table_name, column_name, tables
                    )
                    
                    if not suggestion:
                        self.logger.warning(f"Could not get suggestion for {table_name}.{column_name}")
                        pbar.update(1)
                        continue

                    cache_status = "[CACHED]" if suggestion.get('from_cache', False) else "[AI GENERATED]"
                    print(f"\n\nSuggested foreign key for {table_name}.{column_name}: {cache_status}")
                    print(f"  → {suggestion['suggested_table']}.{suggestion['suggested_column']}")
                    if cache_status == "[AI GENERATED]":
                        print(f"Confidence: {suggestion['confidence']:.2f}")
                        print(f"Reasoning: {suggestion['reasoning']}")
                    print("\nAccept this suggestion? (y/n)", end=" ")
                    
                    response = input().lower()
                    if response == 'y':
                        # Cache the accepted suggestion
                        if not suggestion.get('from_cache', False):
                            cache_key = self._get_cache_key(column_name)
                            self.suggestion_cache[cache_key] = suggestion.copy()
                            self.logger.info(f"Cached suggestion for pattern: {cache_key}")
                        # Write the SQL to the file
                        sql = self.generate_foreign_key_sql(
                            table_name, column_name,
                            suggestion['suggested_table'],
                            suggestion['suggested_column']
                        )
                        self.sql_file.write(sql + "\n")
                        self.sql_file.flush()
                        self.logger.info(f"Added foreign key definition for {table_name}.{column_name}")
                    elif response == 'n':
                        # Cache the rejected suggestion
                        rejection_key = self._get_rejection_cache_key(table_name, column_name)
                        self.rejected_cache[rejection_key] = suggestion.copy()
                        self.logger.info(f"Cached rejected suggestion for {rejection_key}")
                        self.logger.info(f"Skipped {table_name}.{column_name}")
                    
                    pbar.update(1)

        except Exception as e:
            self.logger.error(f"Error during analysis: {str(e)}")
            self.logger.error(traceback.format_exc())
            raise
        finally:
            self.logger.info(f"Total OpenAI API usage: {self.total_tokens} tokens")
            if self.sql_file:
                self.sql_file.close()

    def cleanup(self):
        """Clean up database connections"""
        if self.connection:
            self.connection.close()
        if self.engine:
            self.engine.dispose()
        if self.sql_file and not self.sql_file.closed:
            self.sql_file.close()

def main():
    try:
        # Load configuration and setup logging
        config = load_config()
        logger = setup_logging(config['LOG_LEVEL'])

        # Create and run the foreign key analyzer
        print("Starting foreign key analysis...")
        analyzer = ForeignKeyAnalyzer(config, logger)
        
        try:
            if analyzer.connect_to_db():
                analyzer.analyze_and_suggest()
        finally:
            analyzer.cleanup()

    except ForeignKeyAnalyzerError as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main() 