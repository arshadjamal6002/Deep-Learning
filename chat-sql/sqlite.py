import sqlite3

# connect to sqlite
connection = sqlite3.connect("student.db")

# cursor object
cursor = connection.cursor()

# create table
table_info = '''
CREATE TABLE student (
    name VARCHAR(25),
    class VARCHAR(25),
    section VARCHAR(25),
    marks INTEGER
)
'''

cursor.execute(table_info)

# Insert records

# Insert records into the student table
cursor.execute("INSERT INTO student VALUES ('Alice', '10', 'A', 85)")
cursor.execute("INSERT INTO student VALUES ('Bob', '10', 'B', 78)")
cursor.execute("INSERT INTO student VALUES ('Charlie', '10', 'A', 92)")
cursor.execute("INSERT INTO student VALUES ('David', '10', 'C', 66)")
cursor.execute("INSERT INTO student VALUES ('Eva', '10', 'B', 74)")
cursor.execute("INSERT INTO student VALUES ('Frank', '10', 'A', 88)")
cursor.execute("INSERT INTO student VALUES ('Grace', '10', 'C', 80)")
cursor.execute("INSERT INTO student VALUES ('Hannah', '10', 'A', 90)")
cursor.execute("INSERT INTO student VALUES ('Ian', '10', 'B', 82)")
cursor.execute("INSERT INTO student VALUES ('Jane', '10', 'C', 79)")
cursor.execute("INSERT INTO student VALUES ('Kyle', '10', 'A', 95)")
cursor.execute("INSERT INTO student VALUES ('Laura', '10', 'B', 77)")
cursor.execute("INSERT INTO student VALUES ('Mike', '10', 'C', 69)")
cursor.execute("INSERT INTO student VALUES ('Nina', '10', 'A', 87)")
cursor.execute("INSERT INTO student VALUES ('Oscar', '10', 'B', 91)")
cursor.execute("INSERT INTO student VALUES ('Pam', '10', 'C', 73)")
cursor.execute("INSERT INTO student VALUES ('Quinn', '10', 'A', 89)")
cursor.execute("INSERT INTO student VALUES ('Ray', '10', 'B', 76)")
cursor.execute("INSERT INTO student VALUES ('Sophia', '10', 'C', 93)")
cursor.execute("INSERT INTO student VALUES ('Tom', '10', 'A', 84)")


# Display
print("The records are ")
data = cursor.execute('''SELECT * FROM student''')
for row in data:
    print(row)


# Commit changes in db
connection.commit()
connection.close()