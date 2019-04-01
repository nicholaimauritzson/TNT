#http://patorjk.com/software/taag/#p=display&f=ANSI%20Shadow
# ----------------------------------------------------------
# ██╗   ██╗████████╗██╗██╗     ██╗████████╗██╗   ██╗
# ██║   ██║╚══██╔══╝██║██║     ██║╚══██╔══╝╚██╗ ██╔╝
# ██║   ██║   ██║   ██║██║     ██║   ██║    ╚████╔╝ 
# ██║   ██║   ██║   ██║██║     ██║   ██║     ╚██╔╝  
# ╚██████╔╝   ██║   ██║███████╗██║   ██║      ██║   
#  ╚═════╝    ╚═╝   ╚═╝╚══════╝╚═╝   ╚═╝      ╚═╝   
#           A library of utility methods.
#       
#   Author: Nicholai Mauritzson 2019-...
#           nicholai.mauritzson@nuclear.lu.se
# ----------------------------------------------------------

def printFormatting(title, descriptions, values, errors=None, unit=('Units missing!'):
    """
    Method which prints information to console in a nice way.
    - 'title'..........String containing desired title of the print-out.
    - 'descritpions'...List of strings containing the descriptions of each line to be printed. description=('variable1, varible2, ...).
    - 'values'.........List of variables for each description. value=(val1, val2, ...).
    - 'errors'.........List of errors for each variable (optional). errors=(err1, err2, ...).
    - 'units'..........List of strings containing the unit of each variable. units=(unit1, unit2, ...).
    """
    numEnt = len(descriptions)
    str_len = []
    dots = []

    for i in range(numEnt):
        str_len.append(len(descriptions[i]))

    for i in range(numEnt):
        dots.append(str_len[i]*'.')
    max_dots = len(max(dots, key=len))

    print_dots=[]
    for i in range(numEnt):
        print_dots.append((max_dots-str_len[i]+5)*'.')
        
    print()#Create vertical empty space in terminal
    print('______________________________________________________________________') 
    print('<<<<< %s >>>>>'% title) #Print title
    if errors is not None:
        for i in range(numEnt):
            print('%s%s%.4f (+/-%.4f %s)'%(descriptions[i], print_dots[i], values[i], errors[i], units[i]))
            
    print('______________________________________________________________________')
    print()#Create vertical empty space in terminal