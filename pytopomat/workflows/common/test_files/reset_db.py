from atomate.vasp.database import VaspCalcDb

x = VaspCalcDb.from_db_file("db.json")
x.reset()

print("SUCCESS")