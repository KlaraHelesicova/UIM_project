import csv


def nacteni_dat(data):

    with open(data, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ', quotechar='|') #nejsou oddělovače, nevím jak to napsat
   

data = 'dataSepsis.csv'