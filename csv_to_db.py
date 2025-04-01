import os
import sys
import glob
import re
import logging
from datetime import datetime
from typing import List, Dict, Set
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.exc import SQLAlchemyError
from dotenv import load_dotenv
from tqdm import tqdm
from logging_utils import setup_logging

class CSVImportError(Exception):
    """Base exception class for CSV import errors"""
    pass

class ConfigurationError(CSVImportError):
    """Raised when there are configuration issues"""
    pass

def load_config() -> Dict:
    """Load configuration from .env file"""
    load_dotenv()
    
    config = {}
    
    # Required variables
    required_vars = ['DB_CONNECTION_STRING']

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
    return config

def get_csv_files(folder_path: str, logger: logging.Logger) -> Dict[str, List[str]]:
    """Get all CSV files from the folder and group them by table name."""
    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
    table_files: Dict[str, List[str]] = {}
    
    logger.info(f"Found {len(csv_files)} CSV files in {folder_path}")
    
    for file_path in csv_files:
        # Extract base name without extension and number
        base_name = os.path.basename(file_path)
        table_name = re.sub(r'\d+\.csv$', '.csv', base_name)
        table_name = os.path.splitext(table_name)[0]
        
        if table_name not in table_files:
            table_files[table_name] = []
        table_files[table_name].append(file_path)
    
    # Sort files numerically within each table group
    for table_name in table_files:
        table_files[table_name].sort(key=lambda x: int(re.search(r'\d+', os.path.basename(x)).group()) if re.search(r'\d+', os.path.basename(x)) else 0)
    
    logger.info(f"Grouped files into {len(table_files)} tables")
    return table_files

def get_table_dependencies(engine, table_names: List[str], logger: logging.Logger) -> Dict[str, Set[str]]:
    """Get foreign key dependencies for each table."""
    inspector = inspect(engine)
    dependencies: Dict[str, Set[str]] = {}
    
    for table_name in table_names:
        dependencies[table_name] = set()
        foreign_keys = inspector.get_foreign_keys(table_name)
        for fk in foreign_keys:
            dependencies[table_name].add(fk['referred_table'])
        logger.debug(f"Table {table_name} depends on: {dependencies[table_name]}")
    
    return dependencies

def get_insert_order(dependencies: Dict[str, Set[str]], logger: logging.Logger) -> List[str]:
    """Determine the order of table inserts based on dependencies."""
    # Simple topological sort implementation
    visited = set()
    order = []

    try:
        def visit(table: str):
            logger.debug(f"Visiting table: {table}")
            if table in visited:
                return
            visited.add(table)
            
            # Skip dependencies that aren't in our dependencies dict
            # (these are tables we're not inserting data into)
            for dep in dependencies.get(table, set()):
                if dep in dependencies:
                    visit(dep)
                else:
                    logger.debug(f"Skipping dependency {dep} as it's not in our insert list")
            
            order.append(table)
        
        for table in dependencies:
            if table not in visited:
                visit(table)

    except Exception as e:
        raise CSVImportError(f"Error determining insert order: {str(e)}")
    
    logger.info(f"Determined insert order: {order}")
    return order

def create_output_dir(logger: logging.Logger) -> Path:
    """Create output directory if it doesn't exist."""
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    logger.info(f"Created/verified output directory: {output_dir}")
    return output_dir

def process_csv_files(engine, folder_path: str, logger: logging.Logger):
    """Process CSV files and insert data into the database."""
    # Get all CSV files grouped by table name
    table_files = get_csv_files(folder_path, logger)
    table_names = list(table_files.keys())
    
    # Get table dependencies
    dependencies = get_table_dependencies(engine, table_names, logger)
    
    # Determine insert order
    insert_order = get_insert_order(dependencies, logger)
    
    # Create output directory and SQL file
    output_dir = create_output_dir(logger)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    sql_file = output_dir / f"csv_to_db_{timestamp}.sql"
    
    with open(sql_file, 'w') as f:
        # Start a transaction
        with engine.begin() as connection:
            # Process each table in the correct order
            for table_name in tqdm(insert_order, desc="Processing tables", unit="table"):
                files = table_files[table_name]
                logger.info(f"Processing table {table_name} with {len(files)} files")
                
                for file_path in files:
                    try:
                        # Read CSV file
                        logger.debug(f"Reading CSV file: {file_path}")
                        df = pd.read_csv(file_path)
                        logger.debug(f"Found {len(df)} rows in {file_path}")
                        
                        # Get column information
                        inspector = inspect(engine)
                        columns = inspector.get_columns(table_name)
                        column_info = {col['name']: col for col in columns}
                        
                        # Generate and write INSERT statements
                        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Inserting {os.path.basename(file_path)}", unit="row", leave=False):
                            try:
                                # Convert row to dict and handle NULL values
                                row_dict = row.to_dict()
                                for key, value in row_dict.items():
                                    if pd.isna(value) or value == '':
                                        # Check if column is nullable
                                        if key in column_info and column_info[key].get('nullable', True):
                                            row_dict[key] = None
                                        else:
                                            # For non-nullable columns, use appropriate default values
                                            if isinstance(value, (int, float)):
                                                row_dict[key] = 0
                                            elif isinstance(value, bool):
                                                row_dict[key] = False
                                            else:
                                                row_dict[key] = ''
                                    elif value == 'NULL':
                                        if key in column_info and column_info[key].get('nullable', True):
                                            row_dict[key] = None
                                        else:
                                            # For non-nullable columns, use appropriate default values
                                            if isinstance(value, (int, float)):
                                                row_dict[key] = 0
                                            elif isinstance(value, bool):
                                                row_dict[key] = False
                                            else:
                                                row_dict[key] = ''
                                
                                # Create the INSERT statement using SQLAlchemy's text() with parameters
                                insert_stmt = text(f"INSERT INTO {table_name} ({', '.join(row_dict.keys())}) VALUES ({', '.join(':' + k for k in row_dict.keys())})")
                                
                                # Write to SQL file with actual values
                                values_str = ', '.join(
                                    f"'{str(v)}'" if isinstance(v, str) and v is not None
                                    else 'NULL' if v is None
                                    else str(v)
                                    for v in row_dict.values()
                                )
                                f.write(f"INSERT INTO {table_name} ({', '.join(row_dict.keys())}) VALUES ({values_str});\n")
                                
                                # Execute with parameters
                                connection.execute(insert_stmt, row_dict)
                                
                            except SQLAlchemyError as e:
                                error_msg = f"Error inserting row into {table_name}: {str(e)}"
                                raise CSVImportError(error_msg)
                        
                        logger.info(f"Successfully processed {file_path}")
                        
                    except Exception as e:
                        error_msg = f"Error processing {file_path}: {str(e)}"
                        logger.error(error_msg)
                        raise CSVImportError(error_msg)

def main():
    try:
        # Load configuration and setup logging
        config = load_config()
        logger = setup_logging('csv_to_db', config['LOG_LEVEL'])
        
        if len(sys.argv) != 2:
            print("Usage: python csv_to_db.py <folder_path>")
            sys.exit(1)
        
        folder_path = sys.argv[1]
        if not os.path.isdir(folder_path):
            error_msg = f"Error: {folder_path} is not a valid directory"
            logger.error(error_msg)
            print(error_msg)
            sys.exit(1)
        
        try:
            # Create database engine
            logger.info("Connecting to database...")
            engine = create_engine(config['DB_CONNECTION_STRING'])
            
            # Process CSV files
            logger.info(f"Starting CSV import from {folder_path}")
            process_csv_files(engine, folder_path, logger)
            
            logger.info("CSV import completed successfully!")
            print("CSV import completed successfully!")
            
        except SQLAlchemyError as e:
            error_msg = f"Database error: {str(e)}"
            logger.error(error_msg)
            sys.exit(1)
        finally:
            engine.dispose()
            
    except ConfigurationError as e:
        logger.error(f"Configuration error: {str(e)}")
        sys.exit(1)
    except CSVImportError as e:
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 