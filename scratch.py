import sqlite3

db = sqlite3.connect('/Users/mikewalmsley/repos/db.db')
cursor = db.cursor()

n_rows = 10

cursor.execute(
    '''
    SELECT * FROM catalog
    '''
)
for row in cursor.fetchmany(n_rows):
    print(row)

cursor.execute(
    '''
    SELECT * FROM shardindex
    '''
)
for row in cursor.fetchmany(n_rows):
    print(row)


cursor.execute(
    '''
    SELECT * FROM acquisitions
    '''
)
for row in cursor.fetchmany(n_rows):
    print(row)
