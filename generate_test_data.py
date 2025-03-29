import argparse
import logging
import os
import sys
import traceback
import json
from datetime import datetime

from sqlalchemy import create_engine, text, inspect
from sqlalchemy.exc import SQLAlchemyError
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm
from logging_utils import setup_logging

from database_analyzer import DatabaseDependencyAnalyzer

"""
This script generates test data for a database in the correct dependency order.
It uses the DatabaseDependencyAnalyzer to get the correct order of tables to insert
and then generates test data using OpenAI's API. Finally, it inserts the generated
data into the database.
"""

def load_config():
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
    config['TABLES'] = [t.strip() for t in os.getenv('TABLES', '').split(',')] if os.getenv('TABLES') else []
    config['MAX_RETRY_ATTEMPTS'] = int(os.getenv('MAX_RETRY_ATTEMPTS', '3'))
    config['DEFAULT_ROWS'] = int(os.getenv('DEFAULT_ROWS', '10'))
    config['ROLLBACK_ON_TABLE_FAILURE'] = os.getenv('ROLLBACK_ON_TABLE_FAILURE', 'true').lower() == 'true'
    config['DB_DESCRIPTION'] = os.getenv('DB_DESCRIPTION')
    config['SAVE_SQL'] = os.getenv('SAVE_SQL', 'true').lower() == 'true'

    return config

class DataGeneratorError(Exception):
    """Base exception class for DataGenerator"""
    pass

class ConfigurationError(DataGeneratorError):
    """Raised when there are configuration issues"""
    pass

class DatabaseConnectionError(DataGeneratorError):
    """Raised when database connection fails"""
    pass

class CircularDependencyError(DataGeneratorError):
    """Raised when circular dependencies are detected"""
    pass

class DataGenerator:
    def __init__(self, config, logger):
        """Initialize the DataGenerator"""
        self.config = config
        self.logger = logger
        self.openai_client = None
        self.engine = None
        self.connection = None
        self.inspector = None
        self.total_tokens = 0
        self.sql_file = None
        if self.config.get('SAVE_SQL'):
            self._setup_sql_file()
    
    def _setup_sql_file(self):
        """Create the output directory if it doesn't exist and open the SQL file."""
        os.makedirs('output', exist_ok=True)
        filename = f'output/test_data_inserts_{datetime.now().strftime("%Y%m%d%H%M%S")}.sql'
        self.sql_file = open(filename, 'w')
        self.logger.info(f"SQL output will be saved to: {filename}")

    def connect_to_db(self):
        """Connect to the database using SQLAlchemy"""
        try:
            self.engine = create_engine(self.config['DB_CONNECTION_STRING'])
            self.connection = self.engine.connect()
            self.inspector = inspect(self.engine)
            return True
        except SQLAlchemyError as e:
            error_msg = f"Error connecting to database: {str(e)}"
            self.logger.error(error_msg)
            raise DatabaseConnectionError(error_msg)

    def get_column_info(self, table: str) -> list:
        """Get detailed column information including foreign key relationships"""
        # Get column information using SQLAlchemy inspector
        columns = []
        for col in self.inspector.get_columns(table):
            column = {
                'name': col['name'],
                'type': str(col['type']),
                'null': 'YES' if col.get('nullable', True) else 'NO',
                'key': 'PRI' if col.get('primary_key', False) else '',
                'default': col.get('default', None),
                'extra': 'auto_increment' if col.get('autoincrement', False) else ''
            }
            columns.append(column)

        # Get foreign key information
        foreign_keys = self.inspector.get_foreign_keys(table)
        
        # Enhance column info with foreign key details
        column_info = []
        for col in columns:
            col_dict = col.copy()
            col_dict['foreign_key'] = None
            
            # Find matching foreign key
            for fk in foreign_keys:
                if col['name'] in fk['constrained_columns']:
                    col_dict['foreign_key'] = {
                        'referenced_table': fk['referred_table'],
                        'referenced_column': fk['referred_columns'][0]
                    }
                    break
            
            column_info.append(col_dict)
        
        return column_info

    def get_existing_values(self, table: str, column: str) -> list:
        """Get existing values from a table column"""
        result = self.connection.execute(text(f"SELECT DISTINCT {column} FROM {table}"))
        return [row[0] for row in result.fetchall()]

    def generate_values_with_ai(self, table: str, columns: list, num_rows: int) -> list:
        """Generate realistic values using OpenAI API with foreign key awareness"""
        self.logger.debug(f"Generating values for table {table} with {num_rows} rows")
        if not self.openai_client:
            self.openai_client = OpenAI()

        # Format the column information for the prompt
        column_info = []
        for col in columns:
            info = f"- {col['name']} ({col['type']})"
            if col['null'] == 'NO':
                info += " [NOT NULL]"
            if col['default']:
                info += f" [DEFAULT {col['default']}]"
            if col['extra'] == 'auto_increment':
                info += " [AUTO_INCREMENT]"
            if col['key'] == 'PRI':
                info += " [PRIMARY KEY]"
            if col['foreign_key']:
                ref_table = col['foreign_key']['referenced_table']
                ref_col = col['foreign_key']['referenced_column']
                existing_values = self.get_existing_values(ref_table, ref_col)
                example_values = str(existing_values[:5]) if existing_values else "[]"
                info += f" (foreign key to {ref_table}.{ref_col}, must use values from: {example_values})"
            column_info.append(info)

        column_info_str = "\n".join(column_info)
        prompt = f"""Generate test data for a table named '{table}'. Generate {num_rows} rows, or fewer if the table has many columns to avoid exceeding token limits.
Table columns:
{column_info_str}

Return a JSON object with a 'rows' array containing the generated data. Each row should be an object with column names as keys.
Make the values realistic and contextual to the column names. Only include columns that need values (skip auto-increment primary keys).
For foreign key columns, ONLY use values provided in the examples. If no values are provided, use null. Do not make up values for foreign keys unless the column is not nullable and doesn't have a foreign key constraint.
For numeric values, do not use leading zeros as they are interpreted as octal numbers.
Response format example:
{{
    "rows": [
        {{"column1": "value1", "column2": "value2"}},
        {{"column1": "value3", "column2": "value4"}}
    ]
}}"""

        # Simple schema that ensures we get an array of objects
        schema = {
            "name": "test_data_response",
            "description": "A response containing an array of test data rows",
            "schema": {
                "type": "object",
                "properties": {
                    "rows": {
                        "type": "array",
                        "items": {"type": "object"}
                    }
                },
                "required": ["rows"]
            }
        }

        try:
            self.logger.debug(f"Sending prompt to OpenAI for table {table}:")
            self.logger.debug(prompt)

            system_message = "You are a helpful assistant that generates realistic test data in JSON format. Only respond with the JSON array, no additional text."
            if self.config['DB_DESCRIPTION']:
                system_message = f"You are a helpful assistant that generates realistic test data in JSON format. The database is described as: {self.config['DB_DESCRIPTION']}. Only respond with the JSON array, no additional text."

            response = self.openai_client.chat.completions.create(
                model=self.config['AI_MODEL'],
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt}
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": schema
                },
                temperature=0.7
            )

            usage_stats = response.usage.total_tokens
            self.total_tokens += usage_stats
            self.logger.debug(f"OpenAI API usage stats - {response.usage}")

            response_content = response.choices[0].message.content.strip()
            self.logger.debug(f"Received response from OpenAI for table {table}:")
            self.logger.debug(response_content)

            try:
                generated_data = json.loads(response_content)
                rows = generated_data["rows"]
                if len(rows) < num_rows:
                    self.logger.warning(f"Generated {len(rows)} rows instead of {num_rows} rows for table {table}")
                return rows
            except json.JSONDecodeError as e:
                self.logger.warning(f"Failed to parse JSON response for table {table}: {e}")
            except Exception as e:
                self.logger.warning(f"Error processing response for table {table}: {e}")

        except Exception as e:
            self.logger.warning(f"Error in OpenAI API call for table {table}: {e}")

        return []

    def generate_test_data(self, tables: list, num_rows: int, progress_callback=None):
        """Main method to generate and insert test data for tables"""
        # Create database analyzer
        analyzer = DatabaseDependencyAnalyzer(self.config['DB_CONNECTION_STRING'])
        analyzer.analyze()

        # Get analysis report
        report = analyzer.get_analysis_report()
        self.logger.debug("Database Analysis Report:")
        self.logger.debug(f"Total Tables: {report['total_tables']}")
        self.logger.debug(f"Root Tables: {report['root_tables']}")
        self.logger.debug(f"Leaf Tables: {report['leaf_tables']}")

        # Check for circular dependencies
        if report['cycles']:
            error_msg = "Circular dependencies detected!"
            self.logger.error(error_msg)
            self.logger.error("The following circular dependencies must be resolved:")
            for cycle in report['cycles']:
                self.logger.error(f"  {' -> '.join(cycle)} -> {cycle[0]}")
            raise CircularDependencyError(error_msg)

        # Get ordered list of tables
        ordered_tables = report['insertion_order']
        self.logger.debug(f"Dependency-ordered tables: {ordered_tables}")

        # Filter tables if specific ones were requested
        if tables:
            ordered_tables = [t for t in ordered_tables if t in tables]
            missing_tables = set(tables) - set(ordered_tables)
            if missing_tables:
                self.logger.warning(f"The following requested tables were not found: {', '.join(missing_tables)}")

        self.logger.debug(f"Generating test data in the following order: {', '.join(f'{i}. {table}' for i, table in enumerate(ordered_tables, 1))}")

        total_tables = len(ordered_tables)
        
        # Generate and insert data for each table
        for index, table in enumerate(ordered_tables, 1):
            if progress_callback:
                progress_callback(index, total_tables, f"Processing table: {table}")
            
            self.logger.info(f"Processing table: {table}")

            # Get column information
            columns = self.get_column_info(table)
            self.logger.debug(f"Column information for {table}:")
            for col in columns:
                self.logger.debug(f"  {col}")

            max_attempts = self.config['MAX_RETRY_ATTEMPTS']
            attempt = 0
            rows_inserted = 0
            table_succeeded = True

            while rows_inserted < num_rows and attempt < max_attempts:
                attempt += 1
                rows_needed = num_rows - rows_inserted
                
                # Generate data for remaining rows
                generated_rows = self.generate_values_with_ai(table, columns, rows_needed)

                if not generated_rows:
                    self.logger.error(f"Failed to generate data for table {table}")
                    continue # try again

                # Log generated data
                self.logger.debug(f"Generated data for {table} (attempt {attempt}/{max_attempts}):")
                for row in generated_rows:
                    self.logger.debug(f"  {row}")

                # Insert the generated data
                for row_data in generated_rows:
                    try:
                        # Create the insert SQL
                        columns_str = ', '.join(list(row_data.keys()))
                        values = list(row_data.values())
                        placeholders = ', '.join(':' + str(i) for i in range(len(values)))
                        insert_sql = f"INSERT INTO {table} ({columns_str}) VALUES ({placeholders})"
                        
                        # Insert the data into the database
                        params = {str(i): value for i, value in enumerate(values)}
                        self.connection.execute(text(insert_sql), params)
                        
                        # Save insert into file if enabled
                        if self.config['SAVE_SQL']:
                            # Create a compiled query with literal values for logging
                            compiled_sql = text(insert_sql).bindparams(**params)
                            literal_sql = str(compiled_sql.compile(
                                self.engine,
                                compile_kwargs={"literal_binds": True}
                            ))
                            self.sql_file.write(literal_sql + ";\n")

                        rows_inserted += 1
                    except SQLAlchemyError as e:
                        self.logger.error(f"Failed to insert row into {table}: {str(e)}")
                        continue # try again
                    except Exception as e:
                        self.logger.error(f"Unexpected error inserting row into {table}: {str(e)}")
                        self.logger.error(e)
                        continue # try again

                if rows_inserted < num_rows:
                    self.logger.info(f"Only inserted {rows_inserted}/{num_rows} rows for {table}. Attempting to generate more data (attempt {attempt}/{max_attempts})")

            if rows_inserted == 0:
                self.logger.error(f"Failed to insert any rows for table {table} after {max_attempts} attempts")
                table_succeeded = False
            elif rows_inserted < num_rows:
                self.logger.warning(f"Could only insert {rows_inserted}/{num_rows} rows for table {table} after {max_attempts} attempts")
            else:
                self.logger.info(f"Successfully generated {rows_inserted} rows for table {table}")

            # If a table failed and rollback is enabled, stop processing
            if not table_succeeded and self.config['ROLLBACK_ON_TABLE_FAILURE']:
                self.logger.warning("Stopping further table processing due to failure and ROLLBACK_ON_TABLE_FAILURE=true")
                return False

        self.logger.info("Test data generation complete!")
        return True

    def cleanup(self):
        """Clean up database connections"""
        if self.connection:
            self.connection.close()
        if self.engine:
            self.engine.dispose()
        if self.sql_file:
            self.sql_file.close()

    def run(self, tables=None, num_rows=10, progress_callback=None):
        """Main entry point to run the test data generator"""
        try:
            if self.connect_to_db():
                try:
                    with self.connection.begin():
                        tables_to_process = tables if tables is not None else self.config['TABLES']
                        success = self.generate_test_data(tables_to_process, num_rows, progress_callback)
                        if not success and self.config['ROLLBACK_ON_TABLE_FAILURE']:
                            raise DataGeneratorError("One or more tables failed to generate data completely")
                except Exception as e:
                    self.logger.error(f"Error during test data generation: {str(e)}")
                    self.logger.error("Stack trace:")
                    self.logger.error(traceback.format_exc())
                    raise
        finally:
            self.logger.info(f"Total OpenAI API usage for this run: {self.total_tokens} tokens")
            self.cleanup()

def main():
    try:
        # Load configuration and setup logging
        config = load_config()
        logger = setup_logging('test_data_generator', config['LOG_LEVEL'])

        # Parse command line arguments
        parser = argparse.ArgumentParser(description='Generate test data in correct dependency order')
        parser.add_argument('--tables', help='Comma-separated list of tables (overrides .env)')
        parser.add_argument('--rows', type=int, help=f'Number of rows to generate per table (overrides .env; default: 10)')
        args = parser.parse_args()

        # Create progress bar for CLI
        progress_bar = None
        def update_progress(current, total, message):
            nonlocal progress_bar
            if progress_bar is None:
                progress_bar = tqdm(total=total, desc="Generating test data", unit="table")
            progress_bar.set_description(f"Table {current}/{total}: {message}")
            progress_bar.update(1)

        try:
            # Create and run the test data generator
            print("Starting test data generation...")
            generator = DataGenerator(config, logger)
            tables = [t.strip() for t in args.tables.split(',')] if args.tables else None
            num_rows = args.rows if args.rows is not None else config['DEFAULT_ROWS']
            generator.run(tables=tables, num_rows=num_rows, progress_callback=update_progress)
        finally:
            # Close progress bar
            if progress_bar:
                progress_bar.close()

    except DataGeneratorError as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()