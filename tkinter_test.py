from tkinter import *
from PIL import ImageTk, Image
from mstrio import microstrategy

ANIMAL_STATS_CUBE_ID = '13D00F8811E9328F86670080EFA5E658'
ANIMAL_WORDS_CUBE_ID = 'A83DFB8611E932A783C60080EF554555'

# Establish connection to MicroStrategy API
conn = microstrategy.Connection(base_url="https://env-132697.customer.cloud.mic\
rostrategy.com/MicroStrategyLibrary/api", username="hackathon", \
password="m$trhackathon2019", project_name="MicroStrategy Tutorial")
conn.connect()

def get_query(query):
    global consta
    # Check for empty String
    if len(query) == 0:
        return False

    conn.connect()

    # Get first cube
    asc_df = conn.get_cube(cube_id=ANIMAL_STATS_CUBE_ID)
    # Check if any results for query
    res = asc_df.loc[asc_df['Animal'] == query.title()]

    if res.empty:
        print("No results for that query")
        return False
    else:
        diet = res.iloc[0]['Diet']
        lifespan = res.iloc[0]['Lifespan']
        size = res.iloc[0]['Size']
        consta = res.iloc[0]['Conservation Status']
        details_text.set(res.iloc[0]['Details'])
        top_speed = res.iloc[0]['Top Speed']
        weight = res.iloc[0]['Weight']
        habitat_text.set(res.iloc[0]['Habitat'])
        return True

def search_data():
    global consta
    query = searchEntry.get()
    print('Sending query: %s' % query)
    if not get_query(query):
        searchEntry.insert(0, "No results for the query: ")
        return
    # Populate fields
    name.set(query.title())
    frame.grid()
    conservation.grid()
    # Set conservation
    status_EX.grid_remove()
    status_EW.grid_remove()
    status_CR.grid_remove()
    status_EN.grid_remove()
    status_VU.grid_remove()
    status_NT.grid_remove()
    status_LC.grid_remove()
    print(consta)
    if 'extinct' in consta.lower():
        status_EX.grid()
    elif 'extinct in wild' in consta.lower():
        status_EW.grid()
    elif 'critically endangered' in consta.lower():
        status_CR.grid()
    elif 'endangered' in consta.lower():
        status_EN.grid()
    elif 'threatened' or 'vulnerable' in consta.lower():
        status_VU.grid()
    elif 'near threatened' in consta.lower():
        status_NT.grid()
    elif 'least concerned' in consta.lower():
        status_LC.grid()

master = Tk(screenName='BestiARy')
master.title('BestiARy')
path = 'Title.png'
img = ImageTk.PhotoImage(Image.open(path))
title = Label(master, image=img).grid(row=0, column=0, columnspan=2)

master.rowconfigure(0, weight=1)
master.columnconfigure(0, weight=1)

searchEntry = Entry(master, width=50)
searchEntry.grid(row=1, column=0)
Button(master, text='Search', command=search_data).grid(row=1, column=1)

frame = Frame(master)
frame.grid(row=2, column=0, columnspan=2)
name = StringVar('')
common_name = Label(frame, textvariable=name, font=("Helvetica", 18)).grid(row=0, column=1)
habitat_text = StringVar('')
habitatLabel = Message(frame, text="Habitat: ", width=100, font=("Helvetica", 14))
habitatLabel.grid(row=1, column=0)
habitatText = Message(frame, textvariable=habitat_text, width=650)
habitatText.grid(row=1, column=1, columnspan=2)
details_text = StringVar('')
details = Message(frame, textvariable=details_text, width=650)
details.grid(row=2, column=0, columnspan=3)
details.grid_remove()
frame.grid_remove()

consta = ''
conservation = Frame(master)
conservation_status = Message(conservation, width=650, text='Conservation Status', font=("Helvetica", 12))
conservation_status.grid(row=0, column=0, columnspan=7)
conservation.grid(row=3, column=0, columnspan=2)
status_EX = Message(conservation, text='EX', bg='white', font=("Helvetica", 16))
status_EX.grid(row=1, column=0)
status_EX.grid_remove()
status_EW = Message(conservation, text='EW', bg='gray', font=("Helvetica", 16))
status_EW.grid(row=1, column=1)
status_EW.grid_remove()
status_CR = Message(conservation, text='CR', bg='red', font=("Helvetica", 16))
status_CR.grid(row=1, column=2)
status_CR.grid_remove()
status_EN = Message(conservation, text='EN', bg='orange', font=("Helvetica", 16))
status_EN.grid(row=1, column=3)
status_EN.grid_remove()
status_VU = Message(conservation, text='VU', bg='yellow', font=("Helvetica", 16))
status_VU.grid(row=1, column=4)
status_VU.grid_remove()
status_NT = Message(conservation, text='NT', bg='green', font=("Helvetica", 16))
status_NT.grid(row=1, column=5)
status_NT.grid_remove()
status_LC = Message(conservation, text='LC', bg='blue', font=("Helvetica", 16))
status_LC.grid(row=1, column=6)
status_LC.grid_remove()
conservation.grid_remove()

mainloop()
