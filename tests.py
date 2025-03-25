import unittest
from unittest.mock import Mock, patch, MagicMock
import os
import logging
from sqlalchemy.exc import SQLAlchemyError

from generate_test_data import (
    DataGenerator,
    ConfigurationError,
    DatabaseConnectionError,
    DataGeneratorError,
    load_config,
    setup_logging
)

class DataGeneratorTests(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.mock_config = {
            'DB_CONNECTION_STRING': 'mock://connection',
            'OPENAI_API_KEY': 'mock-key',
            'AI_MODEL': 'gpt-4',
            'LOG_LEVEL': 'DEBUG',
            'TABLES': [],
            'MAX_RETRY_ATTEMPTS': 3,
            'DEFAULT_ROWS': 10,
            'ROLLBACK_ON_TABLE_FAILURE': True,
            'DB_DESCRIPTION': None,
            'SAVE_SQL': False
        }
        self.logger = logging.getLogger('test')
        self.generator = DataGenerator(self.mock_config, self.logger)

    def _create_mock_connection(self):
        """Helper method to create a mock connection with context manager support"""
        mock_connection = MagicMock()
        mock_transaction = MagicMock()
        # Configure the context manager behavior
        mock_connection.begin.return_value.__enter__.return_value = mock_transaction
        mock_connection.begin.return_value.__exit__.return_value = None
        return mock_connection, mock_transaction

    @patch('generate_test_data.create_engine')
    def test_connect_to_db_success(self, mock_create_engine):
        """Test successful database connection."""
        mock_engine = Mock()
        mock_connection = Mock()
        mock_inspector = Mock()
        mock_create_engine.return_value = mock_engine
        mock_engine.connect.return_value = mock_connection
        
        with patch('generate_test_data.inspect', return_value=mock_inspector):
            result = self.generator.connect_to_db()
            
        self.assertTrue(result)
        self.assertEqual(self.generator.engine, mock_engine)
        self.assertEqual(self.generator.connection, mock_connection)
        self.assertEqual(self.generator.inspector, mock_inspector)

    @patch('generate_test_data.create_engine')
    def test_connect_to_db_failure(self, mock_create_engine):
        """Test database connection failure."""
        mock_create_engine.side_effect = SQLAlchemyError("Connection failed")
        
        with self.assertRaises(DatabaseConnectionError):
            self.generator.connect_to_db()

    def test_get_column_info(self):
        """Test getting column information."""
        mock_inspector = Mock()
        mock_inspector.get_columns.return_value = [
            {
                'name': 'id',
                'type': 'INTEGER',
                'nullable': False,
                'primary_key': True,
                'autoincrement': True
            },
            {
                'name': 'name',
                'type': 'VARCHAR',
                'nullable': True,
                'default': None
            }
        ]
        mock_inspector.get_foreign_keys.return_value = []
        
        self.generator.inspector = mock_inspector
        result = self.generator.get_column_info('test_table')
        
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]['name'], 'id')
        self.assertEqual(result[0]['key'], 'PRI')
        self.assertEqual(result[0]['extra'], 'auto_increment')

    @patch('generate_test_data.OpenAI')
    def test_generate_values_with_ai(self, mock_openai):
        """Test generating values using OpenAI API."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [
            Mock(message=Mock(content='{"rows": [{"name": "Test", "age": 25}]}'))
        ]
        mock_response.usage = Mock(total_tokens=100)
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client

        columns = [
            {
                'name': 'name',
                'type': 'VARCHAR',
                'null': 'YES',
                'key': '',
                'default': None,
                'extra': '',
                'foreign_key': None
            },
            {
                'name': 'age',
                'type': 'INTEGER',
                'null': 'YES',
                'key': '',
                'default': None,
                'extra': '',
                'foreign_key': None
            }
        ]

        result = self.generator.generate_values_with_ai('test_table', columns, 1)
        
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['name'], 'Test')
        self.assertEqual(result[0]['age'], 25)

    @patch('generate_test_data.DatabaseDependencyAnalyzer')
    def test_generate_test_data(self, mock_analyzer_class):
        """Test the main test data generation process."""
        mock_analyzer = Mock()
        mock_analyzer.get_analysis_report.return_value = {
            'total_tables': 2,
            'root_tables': ['users'],
            'leaf_tables': ['orders'],
            'cycles': [],
            'insertion_order': ['users', 'orders']
        }
        mock_analyzer_class.return_value = mock_analyzer

        self.generator.get_column_info = Mock(return_value=[
            {
                'name': 'id',
                'type': 'INTEGER',
                'null': 'NO',
                'key': 'PRI',
                'default': None,
                'extra': 'auto_increment',
                'foreign_key': None
            }
        ])
        self.generator.generate_values_with_ai = Mock(return_value=[{'id': 1}])
        self.generator.connection = Mock()
        
        result = self.generator.generate_test_data(['users'], 1)
        
        self.assertTrue(result)
        self.generator.generate_values_with_ai.assert_called_once()

    @patch('generate_test_data.load_dotenv')
    def test_load_config_missing_required_vars(self, mock_load_dotenv):
        """Test configuration loading with missing required variables."""
        # Clear all environment variables and ensure required ones are missing
        mock_env = {
            'SOME_OTHER_VAR': 'value',
            'LOG_LEVEL': 'INFO'
        }
        
        with patch.dict(os.environ, mock_env, clear=True):
            with self.assertRaises(ConfigurationError):
                load_config()

    @patch('generate_test_data.load_dotenv')
    def test_load_config_success(self, mock_load_dotenv):
        """Test successful configuration loading."""
        mock_env = {
            'DB_CONNECTION_STRING': 'test://db',
            'OPENAI_API_KEY': 'test-key',
            'AI_MODEL': 'gpt-4',
            'LOG_LEVEL': 'INFO'
        }
        
        with patch.dict(os.environ, mock_env, clear=True):
            config = load_config()
            
        self.assertEqual(config['DB_CONNECTION_STRING'], 'test://db')
        self.assertEqual(config['OPENAI_API_KEY'], 'test-key')
        self.assertEqual(config['LOG_LEVEL'], 'INFO')

    def test_setup_logging(self):
        """Test logging setup."""
        logger = setup_logging('DEBUG')
        self.assertEqual(logger.level, logging.DEBUG)
        self.assertEqual(len(logger.handlers), 2)

    @patch('generate_test_data.DatabaseDependencyAnalyzer')
    def test_rollback_on_table_failure(self, mock_analyzer_class):
        """Test that rollback occurs when a table fails and ROLLBACK_ON_TABLE_FAILURE is True."""
        # Configure the mock analyzer
        mock_analyzer = Mock()
        mock_analyzer.get_analysis_report.return_value = {
            'total_tables': 2,
            'root_tables': ['users', 'orders'],
            'leaf_tables': [],
            'cycles': [],
            'insertion_order': ['users', 'orders']
        }
        mock_analyzer_class.return_value = mock_analyzer

        # Set up the generator with rollback enabled
        self.mock_config['ROLLBACK_ON_TABLE_FAILURE'] = True
        self.mock_config['MAX_RETRY_ATTEMPTS'] = 1  # Set to 1 to avoid multiple retries
        self.generator = DataGenerator(self.mock_config, self.logger)
        
        # Mock database connection and transaction with context manager support
        mock_connection, mock_transaction = self._create_mock_connection()
        self.generator.connection = mock_connection
        self.generator.engine = Mock()

        # Mock column info to return valid columns
        self.generator.get_column_info = Mock(return_value=[
            {
                'name': 'id',
                'type': 'INTEGER',
                'null': 'NO',
                'key': 'PRI',
                'default': None,
                'extra': 'auto_increment',
                'foreign_key': None
            }
        ])

        # Mock generate_values_with_ai to fail for the second table
        def mock_generate_values(table, columns, num_rows):
            if table == 'orders':
                return []  # Fail for orders table
            return [{'id': 1}]  # Succeed for users table
        self.generator.generate_values_with_ai = Mock(side_effect=mock_generate_values)

        # Mock connect_to_db to return True
        self.generator.connect_to_db = Mock(return_value=True)

        # Run the test and verify the error is raised (rollback is handled automatically)
        with self.assertRaisesRegex(DataGeneratorError, "One or more tables failed to generate data completely"):
            self.generator.run(['users', 'orders'], 1)

    @patch('generate_test_data.DatabaseDependencyAnalyzer')
    def test_continue_on_table_failure(self, mock_analyzer_class):
        """Test that processing continues when a table fails and ROLLBACK_ON_TABLE_FAILURE is False."""
        # Configure the mock analyzer
        mock_analyzer = Mock()
        mock_analyzer.get_analysis_report.return_value = {
            'total_tables': 3,
            'root_tables': ['users', 'orders', 'products'],
            'leaf_tables': [],
            'cycles': [],
            'insertion_order': ['users', 'orders', 'products']
        }
        mock_analyzer_class.return_value = mock_analyzer

        # Set up the generator with rollback disabled and only 1 retry attempt
        self.mock_config['ROLLBACK_ON_TABLE_FAILURE'] = False
        self.mock_config['MAX_RETRY_ATTEMPTS'] = 1  # Set to 1 to avoid multiple retries
        self.generator = DataGenerator(self.mock_config, self.logger)
        
        # Mock database connection and transaction with context manager support
        mock_connection, mock_transaction = self._create_mock_connection()
        self.generator.connection = mock_connection
        self.generator.engine = Mock()

        # Mock column info to return valid columns
        self.generator.get_column_info = Mock(return_value=[
            {
                'name': 'id',
                'type': 'INTEGER',
                'null': 'NO',
                'key': 'PRI',
                'default': None,
                'extra': 'auto_increment',
                'foreign_key': None
            }
        ])

        # Mock generate_values_with_ai to fail for the middle table
        def mock_generate_values(table, columns, num_rows):
            if table == 'orders':
                return []  # Fail for orders table
            return [{'id': 1}]  # Succeed for other tables
        self.generator.generate_values_with_ai = Mock(side_effect=mock_generate_values)

        # Mock connect_to_db to return True
        self.generator.connect_to_db = Mock(return_value=True)

        # Run the test through the run method
        self.generator.run(['users', 'orders', 'products'], 1)
        
        # Verify the results
        mock_transaction.rollback.assert_not_called()  # Should not have triggered a rollback
        
        # Verify that generate_values_with_ai was called once for each table
        # (no retries since MAX_RETRY_ATTEMPTS is 1)
        self.assertEqual(self.generator.generate_values_with_ai.call_count, 3)

    @patch('generate_test_data.DatabaseDependencyAnalyzer')
    def test_retry_on_table_failure(self, mock_analyzer_class):
        """Test that a table failing initially but succeeding on retry works correctly."""
        # Configure the mock analyzer
        mock_analyzer = Mock()
        mock_analyzer.get_analysis_report.return_value = {
            'total_tables': 2,
            'root_tables': ['users', 'orders'],
            'leaf_tables': [],
            'cycles': [],
            'insertion_order': ['users', 'orders']
        }
        mock_analyzer_class.return_value = mock_analyzer

        # Set up the generator with 2 retry attempts
        self.mock_config['MAX_RETRY_ATTEMPTS'] = 2
        self.generator = DataGenerator(self.mock_config, self.logger)
        
        # Mock database connection and transaction
        mock_connection, mock_transaction = self._create_mock_connection()
        self.generator.connection = mock_connection
        self.generator.engine = Mock()

        # Mock column info to return valid columns
        self.generator.get_column_info = Mock(return_value=[
            {
                'name': 'id',
                'type': 'INTEGER',
                'null': 'NO',
                'key': 'PRI',
                'default': None,
                'extra': 'auto_increment',
                'foreign_key': None
            }
        ])

        # Mock generate_values_with_ai to fail first time for orders table, succeed second time
        attempts = {'orders': 0}
        def mock_generate_values(table, columns, num_rows):
            if table == 'orders':
                attempts['orders'] += 1
                if attempts['orders'] == 1:
                    return []  # Fail first attempt
                return [{'id': 2}]  # Succeed second attempt
            return [{'id': 1}]  # Always succeed for users table
        self.generator.generate_values_with_ai = Mock(side_effect=mock_generate_values)

        # Mock connect_to_db to return True
        self.generator.connect_to_db = Mock(return_value=True)

        # Run the generator - should not raise any exceptions
        self.generator.run(['users', 'orders'], 1)
        
        # Verify generate_values_with_ai was called correct number of times
        # Once for users table, twice for orders table (fail + success)
        self.assertEqual(self.generator.generate_values_with_ai.call_count, 3)
        
        # Verify the calls to generate_values_with_ai
        calls = self.generator.generate_values_with_ai.call_args_list
        self.assertEqual(calls[0][0][0], 'users')  # First call for users
        self.assertEqual(calls[1][0][0], 'orders')  # First attempt for orders
        self.assertEqual(calls[2][0][0], 'orders')  # Second attempt for orders

if __name__ == '__main__':
    unittest.main() 