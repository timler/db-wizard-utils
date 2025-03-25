import os
from sqlalchemy import create_engine, MetaData, inspect
import networkx as nx
import matplotlib.pyplot as plt
from graphviz import Digraph
from collections import defaultdict
from dotenv import load_dotenv
import sys
import csv

"""
This tool analyzes the dependencies of a database and generates a report
and if you run it as a script, it will generate a visualization of the
dependencies and save it as a PNG file (database_dependencies.png).
"""

class DatabaseDependencyAnalyzer:
    def __init__(self, connection_string):
        """Initialize with database connection string."""
        self.engine = create_engine(connection_string)
        self.metadata = MetaData()
        self.graph = nx.DiGraph()
        self.dependencies = defaultdict(list)

    def analyze(self):
        """Analyze database and build dependency graph."""
        # Reflect database structure
        self.metadata.reflect(bind=self.engine)
        inspector = inspect(self.engine)

        # Build dependency graph
        for table_name in inspector.get_table_names():
            self.graph.add_node(table_name)
            fks = inspector.get_foreign_keys(table_name)

            for fk in fks:
                parent_table = fk['referred_table']
                self.graph.add_edge(parent_table, table_name)
                self.dependencies[table_name].append({
                    'parent_table': parent_table,
                    'constrained_columns': fk['constrained_columns'],
                    'referred_columns': fk['referred_columns']
                })

    def get_insertion_order(self):
        """Get optimal table insertion order."""
        try:
            return list(nx.topological_sort(self.graph))
        except nx.NetworkXUnfeasible:
            return self._handle_circular_dependencies()

    def _handle_circular_dependencies(self):
        """Handle circular dependencies by finding strongly connected components."""
        components = list(nx.strongly_connected_components(self.graph))
        # Create a new graph with components as nodes
        component_graph = nx.DiGraph()

        for i, component in enumerate(components):
            component_graph.add_node(i, tables=component)

        # Add edges between components
        for i, comp1 in enumerate(components):
            for j, comp2 in enumerate(components):
                if i != j:
                    if any(self.graph.has_edge(n1, n2) 
                          for n1 in comp1 for n2 in comp2):
                        component_graph.add_edge(i, j)
        
        # Get component order
        component_order = list(nx.topological_sort(component_graph))

        # Flatten the order
        return [table 
                for idx in component_order 
                for table in components[idx]]

    def visualize(self, output_file='database_dependencies'):
        """Create a visual representation of the dependencies."""
        dot = Digraph(comment='Database Dependencies')
        dot.attr(rankdir='LR')

        # Add nodes
        for node in self.graph.nodes():
            dot.node(node, node)

        # Add edges
        for edge in self.graph.edges():
            dot.edge(edge[0], edge[1])

        # Save the visualization
        os.makedirs('output', exist_ok=True)
        dot.render(os.path.join('output', output_file), format='png', cleanup=True)

    def get_analysis_report(self):
        """Generate a comprehensive analysis report."""
        report = {
            'total_tables': len(self.graph.nodes()),
            'root_tables': [n for n in self.graph.nodes() if self.graph.in_degree(n) == 0],
            'leaf_tables': [n for n in self.graph.nodes() if self.graph.out_degree(n) == 0],
            'cycles': list(nx.simple_cycles(self.graph)),
            'insertion_order': self.get_insertion_order(),
            'table_dependencies': dict(self.dependencies)
        }
        return report

    def print_report(self):
        """Print a formatted analysis report."""
        report = self.get_analysis_report()

        print("\n=== Database Dependency Analysis Report ===")
        print(f"\nTotal Tables: {report['total_tables']}")

        print("\nRoot Tables (No dependencies):")
        for table in report['root_tables']:
            print(f"  - {table}")

        print("\nLeaf Tables (No dependents):")
        for table in report['leaf_tables']:
            print(f"  - {table}")

        print("\nCircular Dependencies:")
        if report['cycles']:
            for cycle in report['cycles']:
                print(f"  - {' -> '.join(cycle)} -> {cycle[0]}")
        else:
            print("  No circular dependencies found")

        print("\nRecommended Insertion Order:")
        for i, table in enumerate(report['insertion_order'], 1):
            print(f"  {i}. {table}")

        print("\nDetailed Dependencies:")
        for table, deps in report['table_dependencies'].items():
            if deps:
                print(f"\n  {table}:")
                for dep in deps:
                    print(f"    Depends on: {dep['parent_table']}")
                    print(f"      Columns: {dep['constrained_columns']} -> {dep['referred_columns']}")

def main():
    # get the connection string from the environment variable
    load_dotenv()
    connection_string = os.getenv('DB_CONNECTION_STRING')
    if not connection_string:
        print("Error: DB_CONNECTION_STRING environment variable is required", file=sys.stderr)
        sys.exit(1)

    try:
        # Initialize analyzer
        analyzer = DatabaseDependencyAnalyzer(connection_string)
        analyzer.analyze()

        # Generate and save visualization
        analyzer.visualize()

        # Print analysis report
        analyzer.print_report()

    except Exception as e:
        print(f"Error during analysis: {str(e)}")

if __name__ == "__main__":
    main()