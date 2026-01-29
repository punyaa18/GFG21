"""
Day 15: SQL Integration with Python and Databases
Demonstrates SQL operations, database interactions, and data visualization from databases.
"""

import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (14, 8)
import os
os.makedirs('outputs', exist_ok=True)

# ========================================
# Step 1: Create SQLite Database
# ========================================
db_file = 'outputs/company_database.db'

# Remove existing database
if os.path.exists(db_file):
    os.remove(db_file)

conn = sqlite3.connect(db_file)
cursor = conn.cursor()

print("Creating database schema...")

# Create tables
cursor.execute('''
    CREATE TABLE employees (
        employee_id INTEGER PRIMARY KEY,
        name TEXT NOT NULL,
        department TEXT NOT NULL,
        salary REAL NOT NULL,
        hire_date TEXT NOT NULL,
        age INTEGER NOT NULL
    )
''')

cursor.execute('''
    CREATE TABLE sales (
        sale_id INTEGER PRIMARY KEY,
        employee_id INTEGER,
        amount REAL,
        sale_date TEXT,
        region TEXT,
        FOREIGN KEY (employee_id) REFERENCES employees(employee_id)
    )
''')

cursor.execute('''
    CREATE TABLE projects (
        project_id INTEGER PRIMARY KEY,
        name TEXT,
        budget REAL,
        status TEXT,
        start_date TEXT,
        end_date TEXT
    )
''')

# ========================================
# Step 2: Insert Sample Data
# ========================================
print("Inserting sample data...")

employees_data = [
    (1, 'Alice Johnson', 'Sales', 75000, '2020-01-15', 35),
    (2, 'Bob Smith', 'Engineering', 95000, '2019-03-20', 42),
    (3, 'Carol Williams', 'Sales', 72000, '2020-06-10', 38),
    (4, 'David Brown', 'Engineering', 98000, '2018-11-05', 45),
    (5, 'Eve Davis', 'Marketing', 68000, '2021-02-14', 32),
    (6, 'Frank Miller', 'Sales', 78000, '2020-08-22', 40),
    (7, 'Grace Lee', 'Engineering', 102000, '2017-05-10', 48),
    (8, 'Henry Wilson', 'HR', 65000, '2019-09-30', 36),
    (9, 'Iris Taylor', 'Marketing', 71000, '2020-12-01', 34),
    (10, 'Jack Anderson', 'Sales', 80000, '2019-07-15', 41),
]

cursor.executemany('INSERT INTO employees VALUES (?, ?, ?, ?, ?, ?)', employees_data)

# Generate sales data
np.random.seed(42)
sales_data = []
for i in range(100):
    employee_id = np.random.choice(range(1, 11))
    amount = np.random.uniform(1000, 50000)
    sale_date = f'2023-{np.random.randint(1,13):02d}-{np.random.randint(1,29):02d}'
    region = np.random.choice(['North', 'South', 'East', 'West'])
    sales_data.append((i+1, employee_id, amount, sale_date, region))

cursor.executemany('INSERT INTO sales VALUES (?, ?, ?, ?, ?)', sales_data)

# Insert projects
projects_data = [
    (1, 'Website Redesign', 50000, 'Completed', '2023-01-10', '2023-06-15'),
    (2, 'Mobile App', 120000, 'In Progress', '2023-03-01', '2024-03-01'),
    (3, 'Cloud Migration', 80000, 'In Progress', '2023-05-20', '2024-01-30'),
    (4, 'Data Analytics', 60000, 'Planning', '2023-11-01', '2024-05-01'),
    (5, 'Security Upgrade', 40000, 'Completed', '2023-02-15', '2023-04-30'),
]

cursor.executemany('INSERT INTO projects VALUES (?, ?, ?, ?, ?, ?)', projects_data)

conn.commit()
print("Database created and populated!")

# ========================================
# Step 3: SQL Queries and Data Retrieval
# ========================================
print("\n=== SQL QUERY RESULTS ===")

# Query 1: All employees
query1 = "SELECT * FROM employees"
df_employees = pd.read_sql_query(query1, conn)
print("\n1. All Employees:")
print(df_employees)

# Query 2: Employees by department
query2 = "SELECT department, COUNT(*) as count, AVG(salary) as avg_salary FROM employees GROUP BY department"
df_dept = pd.read_sql_query(query2, conn)
print("\n2. Employees by Department:")
print(df_dept)

# Query 3: Sales data
query3 = "SELECT * FROM sales LIMIT 10"
df_sales = pd.read_sql_query(query3, conn)
print("\n3. First 10 Sales:")
print(df_sales)

# Query 4: Sales by region
query4 = "SELECT region, COUNT(*) as sales_count, SUM(amount) as total_amount FROM sales GROUP BY region"
df_region = pd.read_sql_query(query4, conn)
print("\n4. Sales by Region:")
print(df_region)

# Query 5: Top performing employees (by sales)
query5 = """
SELECT e.name, COUNT(s.sale_id) as num_sales, SUM(s.amount) as total_sales, AVG(s.amount) as avg_sale
FROM employees e
LEFT JOIN sales s ON e.employee_id = s.employee_id
GROUP BY e.employee_id
ORDER BY total_sales DESC LIMIT 10
"""
df_top_emp = pd.read_sql_query(query5, conn)
print("\n5. Top Performing Employees:")
print(df_top_emp)

# ========================================
# Step 4: Sales Analysis Visualization
# ========================================
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. Sales by Region
df_region_sorted = df_region.sort_values('total_amount', ascending=False)
axes[0, 0].bar(df_region_sorted['region'], df_region_sorted['total_amount'], 
              color='steelblue', edgecolor='black', alpha=0.7)
axes[0, 0].set_ylabel('Total Sales Amount ($)', fontweight='bold')
axes[0, 0].set_title('Total Sales by Region', fontweight='bold', fontsize=12)
axes[0, 0].grid(True, alpha=0.3, axis='y')

# 2. Employee Performance
df_top_emp_sorted = df_top_emp.sort_values('total_sales', ascending=False)
axes[0, 1].barh(df_top_emp_sorted['name'], df_top_emp_sorted['total_sales'], 
               color='green', edgecolor='black', alpha=0.7)
axes[0, 1].set_xlabel('Total Sales ($)', fontweight='bold')
axes[0, 1].set_title('Top 10 Sales Performers', fontweight='bold', fontsize=12)
axes[0, 1].grid(True, alpha=0.3, axis='x')

# 3. Department salary analysis
df_dept_sorted = df_dept.sort_values('avg_salary', ascending=False)
axes[1, 0].bar(df_dept_sorted['department'], df_dept_sorted['avg_salary'], 
              color='orange', edgecolor='black', alpha=0.7)
axes[1, 0].set_ylabel('Average Salary ($)', fontweight='bold')
axes[1, 0].set_title('Average Salary by Department', fontweight='bold', fontsize=12)
axes[1, 0].grid(True, alpha=0.3, axis='y')

# 4. Sales count by region
axes[1, 1].pie(df_region['sales_count'], labels=df_region['region'], autopct='%1.1f%%',
              startangle=90, colors=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A'])
axes[1, 1].set_title('Sales Distribution by Region', fontweight='bold', fontsize=12)

plt.tight_layout()
plt.savefig('outputs/sales_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

print("\nSales analysis plot saved!")

# ========================================
# Step 5: Employee Analysis
# ========================================
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. Salary distribution
axes[0, 0].hist(df_employees['salary'], bins=15, color='steelblue', edgecolor='black', alpha=0.7)
axes[0, 0].set_xlabel('Salary ($)', fontweight='bold')
axes[0, 0].set_ylabel('Frequency', fontweight='bold')
axes[0, 0].set_title('Salary Distribution', fontweight='bold', fontsize=12)
axes[0, 0].grid(True, alpha=0.3, axis='y')

# 2. Age vs Salary
colors_dept = {'Sales': 'red', 'Engineering': 'blue', 'Marketing': 'green', 'HR': 'orange'}
for dept in df_employees['department'].unique():
    dept_data = df_employees[df_employees['department'] == dept]
    axes[0, 1].scatter(dept_data['age'], dept_data['salary'], s=100, label=dept, 
                      color=colors_dept.get(dept, 'gray'), alpha=0.6, edgecolors='black')
axes[0, 1].set_xlabel('Age', fontweight='bold')
axes[0, 1].set_ylabel('Salary ($)', fontweight='bold')
axes[0, 1].set_title('Age vs Salary by Department', fontweight='bold', fontsize=12)
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# 3. Employee count by department
dept_counts = df_employees['department'].value_counts()
axes[1, 0].bar(dept_counts.index, dept_counts.values, color='coral', edgecolor='black', alpha=0.7)
axes[1, 0].set_ylabel('Number of Employees', fontweight='bold')
axes[1, 0].set_title('Employee Count by Department', fontweight='bold', fontsize=12)
axes[1, 0].grid(True, alpha=0.3, axis='y')

# 4. Hire date distribution
hire_dates = pd.to_datetime(df_employees['hire_date'])
hire_year = hire_dates.dt.year.value_counts().sort_index()
axes[1, 1].bar(hire_year.index, hire_year.values, color='purple', edgecolor='black', alpha=0.7)
axes[1, 1].set_xlabel('Year', fontweight='bold')
axes[1, 1].set_ylabel('Employees Hired', fontweight='bold')
axes[1, 1].set_title('Employees Hired by Year', fontweight='bold', fontsize=12)
axes[1, 1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('outputs/employee_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

print("Employee analysis plot saved!")

# ========================================
# Step 6: Transaction Analysis
# ========================================
query_transactions = "SELECT * FROM sales"
df_all_sales = pd.read_sql_query(query_transactions, conn)
df_all_sales['sale_date'] = pd.to_datetime(df_all_sales['sale_date'])

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. Sales amount distribution
axes[0, 0].hist(df_all_sales['amount'], bins=30, color='steelblue', edgecolor='black', alpha=0.7)
axes[0, 0].set_xlabel('Sale Amount ($)', fontweight='bold')
axes[0, 0].set_ylabel('Frequency', fontweight='bold')
axes[0, 0].set_title('Sales Amount Distribution', fontweight='bold', fontsize=12)
axes[0, 0].grid(True, alpha=0.3, axis='y')

# 2. Sales over time
daily_sales = df_all_sales.groupby('sale_date')['amount'].sum().sort_index()
axes[0, 1].plot(daily_sales.index, daily_sales.values, linewidth=2, marker='o', markersize=4, color='green')
axes[0, 1].set_xlabel('Date', fontweight='bold')
axes[0, 1].set_ylabel('Total Sales ($)', fontweight='bold')
axes[0, 1].set_title('Daily Sales Trend', fontweight='bold', fontsize=12)
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].tick_params(axis='x', rotation=45)

# 3. Sales by region
df_region_sorted = df_all_sales.groupby('region')['amount'].sum().sort_values(ascending=False)
axes[1, 0].bar(df_region_sorted.index, df_region_sorted.values, color='orange', edgecolor='black', alpha=0.7)
axes[1, 0].set_ylabel('Total Sales ($)', fontweight='bold')
axes[1, 0].set_title('Total Sales by Region', fontweight='bold', fontsize=12)
axes[1, 0].grid(True, alpha=0.3, axis='y')

# 4. Box plot of sales by region
regions = df_all_sales['region'].unique()
region_data = [df_all_sales[df_all_sales['region'] == r]['amount'].values for r in regions]
bp = axes[1, 1].boxplot(region_data, labels=regions, patch_artist=True)
for patch in bp['boxes']:
    patch.set_facecolor('lightblue')
axes[1, 1].set_ylabel('Sale Amount ($)', fontweight='bold')
axes[1, 1].set_title('Sales Distribution by Region', fontweight='bold', fontsize=12)
axes[1, 1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('outputs/transaction_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

print("Transaction analysis plot saved!")

# ========================================
# Step 7: Advanced SQL Queries
# ========================================
print("\n=== ADVANCED SQL ANALYSIS ===")

# Complex join query
complex_query = """
SELECT 
    e.department,
    COUNT(DISTINCT e.employee_id) as num_employees,
    COUNT(s.sale_id) as total_sales,
    SUM(s.amount) as total_revenue,
    AVG(s.amount) as avg_sale_amount,
    AVG(e.salary) as avg_salary
FROM employees e
LEFT JOIN sales s ON e.employee_id = s.employee_id
GROUP BY e.department
ORDER BY total_revenue DESC
"""

df_complex = pd.read_sql_query(complex_query, conn)
print("\nDepartment Performance Summary:")
print(df_complex.to_string())

# ========================================
# Step 8: Database Summary Statistics
# ========================================
fig, ax = plt.subplots(figsize=(12, 8))

summary_stats = f"""
DATABASE SUMMARY STATISTICS
{'='*60}

EMPLOYEES TABLE:
  Total Employees: {len(df_employees)}
  Average Salary: ${df_employees['salary'].mean():,.2f}
  Salary Range: ${df_employees['salary'].min():,.0f} - ${df_employees['salary'].max():,.0f}
  Average Age: {df_employees['age'].mean():.1f} years
  
SALES TABLE:
  Total Transactions: {len(df_all_sales)}
  Total Sales Revenue: ${df_all_sales['amount'].sum():,.2f}
  Average Sale Amount: ${df_all_sales['amount'].mean():,.2f}
  Largest Sale: ${df_all_sales['amount'].max():,.2f}
  
DEPARTMENTS:
  Number of Departments: {len(df_employees['department'].unique())}
  Departments: {', '.join(df_employees['department'].unique())}
  
REGIONS:
  Number of Sales Regions: {len(df_all_sales['region'].unique())}
  Regions: {', '.join(sorted(df_all_sales['region'].unique()))}
  
PROJECTS TABLE:
  Total Projects: 5
  Completed: 2
  In Progress: 2
  Planning: 1
"""

ax.text(0.05, 0.95, summary_stats, fontsize=11, family='monospace',
       verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
ax.axis('off')
plt.tight_layout()
plt.savefig('outputs/database_summary.png', dpi=300, bbox_inches='tight')
plt.close()

print("Database summary plot saved!")

# Close connection
conn.close()

print("\n✅ SQL Integration Complete!")
print("Generated outputs:")
print("  - outputs/company_database.db (SQLite database)")
print("  - outputs/sales_analysis.png")
print("  - outputs/employee_analysis.png")
print("  - outputs/transaction_analysis.png")
print("  - outputs/database_summary.png")
