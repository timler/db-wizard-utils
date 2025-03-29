import os
import sys
import logging
import json
from datetime import datetime
from typing import Dict
import traceback

from sqlalchemy import create_engine, MetaData, inspect
from sqlalchemy.exc import SQLAlchemyError
from openai import OpenAI
from dotenv import load_dotenv
from tqdm import tqdm
from logging_utils import setup_logging

class TableDescriberError(Exception):
    """Base exception class for TableDescriber"""
    pass

class ConfigurationError(TableDescriberError):
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
    config['DB_DESCRIPTION'] = os.getenv('DB_DESCRIPTION', '')
    config['DB_INSTRUCTIONS'] = os.getenv('DB_INSTRUCTIONS', '')
    
    return config

class TableDescriber:
    def __init__(self, config: Dict, logger: logging.Logger):
        """Initialize the TableDescriber"""
        self.config = config
        self.logger = logger
        self.engine = None
        self.connection = None
        self.inspector = None
        self.metadata = None
        self.openai_client = None
        self.total_tokens = 0
        self.output_file = None
        self._setup_output_file()

    def _setup_output_file(self):
        """Create the output directory and markdown file for table descriptions"""
        os.makedirs('output', exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'output/table_descriptions_{timestamp}.md'
        self.output_file = open(filename, 'w')
        self.logger.info(f"Descriptions will be saved to: {filename}")
        # Write markdown header
        self.output_file.write("# Database Table Descriptions\n\n")
        self.output_file.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

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
            raise TableDescriberError(error_msg)

    def get_table_definition(self, table_name: str) -> Dict:
        """Get detailed information about a table"""
        columns = self.inspector.get_columns(table_name)
        foreign_keys = self.inspector.get_foreign_keys(table_name)
        primary_key = self.inspector.get_pk_constraint(table_name)
        indexes = self.inspector.get_indexes(table_name)
        
        return {
            'columns': columns,
            'foreign_keys': foreign_keys,
            'primary_key': primary_key,
            'indexes': indexes
        }

    def get_table_description(self, table_name: str, table_info: Dict) -> str:
        """Use OpenAI to get a description of the table"""
        if not self.openai_client:
            self.openai_client = OpenAI()

        # Build the context sections only if they're not empty
        context_sections = []
        if self.config['DB_DESCRIPTION']:
            context_sections.extend([
                "You are analyzing a table in the following database:",
                "",
                self.config['DB_DESCRIPTION'],
                ""
            ])
        
        if self.config['DB_INSTRUCTIONS']:
            context_sections.extend([
                "Additional context about the database structure:",
                self.config['DB_INSTRUCTIONS'],
                ""
            ])

        # Prepare the prompt
        prompt = f"""{chr(10).join(context_sections)}Please describe the following table in one or two clear, concise sentences. Focus on its purpose and main relationships.

Table Name: {table_name}

Columns:
{json.dumps([{
    'name': col['name'],
    'type': str(col['type']),
    'nullable': col.get('nullable', True),
    'default': str(col.get('default', 'None')),
} for col in table_info['columns']], indent=2)}

Primary Key: {json.dumps(table_info['primary_key'], indent=2)}

Foreign Keys:
{json.dumps(table_info['foreign_keys'], indent=2)}

Indexes:
{json.dumps(table_info['indexes'], indent=2)}

Your response should be a clear, technical description that a database administrator would find useful. Focus on the table's purpose and its relationships with other tables."""

        try:
            messages = [
                {"role": "system", "content": "You are a database expert providing concise, technical descriptions of database tables. Respond with only the description, no additional formatting or text."},
                {"role": "user", "content": prompt}
            ]

            response = self.openai_client.chat.completions.create(
                model=self.config['AI_MODEL'],
                messages=messages,
                temperature=0.1,
                max_tokens=150  # Limit response length to ensure conciseness
            )

            self.total_tokens += response.usage.total_tokens
            self.logger.debug(f"OpenAI API usage stats for {table_name}: {response.usage}")
            
            return response.choices[0].message.content.strip()

        except Exception as e:
            self.logger.error(f"Error getting description for {table_name}: {e}")
            self.logger.error(traceback.format_exc())
            return f"Error getting description: {str(e)}"

    def describe_tables(self):
        """Main method to describe all tables in the database"""
        try:
            # Get all tables
            tables = self.inspector.get_table_names()
            self.logger.info(f"Found {len(tables)} tables in the database")

            # Create progress bar
            with tqdm(total=len(tables), desc="Describing tables") as pbar:
                for table_name in tables:
                    pbar.set_description(f"Analyzing {table_name}")
                    self.logger.info(f"Getting description for {table_name}")
                    
                    # Get table info
                    table_info = self.get_table_definition(table_name)
                    
                    # Get description from OpenAI
                    description = self.get_table_description(table_name, table_info)
                    
                    # Write to markdown file
                    self.output_file.write(f"## {table_name}\n\n")
                    self.output_file.write(f"{description}\n\n")
                    self.output_file.write("### Schema\n\n")
                    self.output_file.write("```sql\n")
                    # Write column definitions
                    for col in table_info['columns']:
                        nullable = "NULL" if col.get('nullable', True) else "NOT NULL"
                        default = f"DEFAULT {col.get('default', 'NULL')}" if col.get('default') is not None else ""
                        self.output_file.write(f"{col['name']} {col['type']} {nullable} {default}\n")
                    self.output_file.write("```\n\n")

                    # Write foreign keys section
                    if table_info['foreign_keys']:
                        self.output_file.write("### Foreign Keys\n\n")
                        self.output_file.write("| Source Column | Referenced Table | Referenced Column |\n")
                        self.output_file.write("|--------------|-----------------|------------------|\n")
                        for fk in table_info['foreign_keys']:
                            for src_col, ref_col in zip(fk['constrained_columns'], fk['referred_columns']):
                                self.output_file.write(f"| {src_col} | {fk['referred_table']} | {ref_col} |\n")
                        self.output_file.write("\n")

                    # Write potential missing foreign keys section
                    potential_fk_columns = [
                        (col['name'], str(col['type'])) for col in table_info['columns']
                        if col['name'].endswith('_id') and 
                        col['name'] not in [
                            fk_col for fk in table_info['foreign_keys']
                            for fk_col in fk['constrained_columns']
                        ]
                    ]
                    
                    if potential_fk_columns:
                        self.output_file.write("### Potential Missing Foreign Keys\n\n")
                        self.output_file.write("The following columns end with '_id' but are not defined as foreign keys:\n\n")
                        self.output_file.write("| Column | Type |\n")
                        self.output_file.write("|--------|------|\n")
                        for col_name, col_type in potential_fk_columns:
                            self.output_file.write(f"| {col_name} | {col_type} |\n")
                        self.output_file.write("\n")
                    
                    self.output_file.flush()
                    
                    pbar.update(1)

        except Exception as e:
            self.logger.error(f"Error during analysis: {str(e)}")
            self.logger.error(traceback.format_exc())
            raise
        finally:
            self.logger.info(f"Total OpenAI API usage: {self.total_tokens} tokens")
            if self.output_file:
                self.output_file.close()

    def cleanup(self):
        """Clean up database connections"""
        if self.connection:
            self.connection.close()
        if self.engine:
            self.engine.dispose()
        if self.output_file and not self.output_file.closed:
            self.output_file.close()

def main():
    try:
        # Load configuration and setup logging
        config = load_config()
        logger = setup_logging('table_describer', config['LOG_LEVEL'])

        # Create and run the table describer
        print("Starting table description analysis...")
        describer = TableDescriber(config, logger)
        
        try:
            if describer.connect_to_db():
                describer.describe_tables()
        finally:
            describer.cleanup()

    except TableDescriberError as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()