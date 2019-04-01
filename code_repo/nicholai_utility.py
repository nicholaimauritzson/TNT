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

def printFormatting(title, descriptions, values, errors=None, unit='Units missing units!'):
    """
    WIP print formatting function.
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
            print('%s%s%.4f (+/-%.4f %s)'%(descriptions[i], print_dots[i], values[i], errors[i], unit))
            
    print('______________________________________________________________________')
    print()#Create vertical empty space in terminal