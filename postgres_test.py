import train_data
import psycopg2
'''
con = psycopg2.connect(host='localhost', database='IAmHere', user='postgres', password='leds123')
cursor = con.cursor()

sql = "INSERT INTO collection (nome, path) VALUES ('%s','%s')" % ("IFES", "IFES")
cursor.execute(sql)

sql = "INSERT INTO collection (nome, path) VALUES ('%s','%s')" % ("Serra", "IFES.Serra")
cursor.execute(sql)
con.commit()

train_data.get_data("","IFES.Serra.Alunos",True)
'''

X, y = train_data.get_data_from_db("IFES.Serra.Alunos")

print(y)