'''
SlapJack 

Rules: 

52 deck card excluding the Jokers

Numbers = continue 

When you see a Jack - Slap it!! 
'''
import time 
import datetime
import random 
import threading # Limited input statement
import fontstyle # changes the fonts 

# Define the cards #
print(" ")
print("_____Welcome to SlapJack_______")
time.sleep(3)
print(" ")
print("For those who don't know how to play:")
time.sleep(3)
print("If you see a Jack played, you slap it!! ")
time.sleep(3)
print("If you see a double played, you slap it!! \n(2 of the same card)")
time.sleep(3)
print("If you see a marraige, slap it!! \n(A queen and King together)")
time.sleep(3)
print("You play until 1 person has all the cards!! \nThat person is the winner")
time.sleep(3)
print("________________________")
time.sleep(3)
print("In order to slap a card, all you do is click ENTER")
time.sleep(3)
print("The game goes quickly so pay attention")

def play_cards(c):
    # creates list 
    num_cards = range(1,c)
    # convert range to list 
    cards = list(num_cards)
    # print(deck)
    # creates face_cards 
    face_cards = ['Jack', 'Queen', 'King', 'Ace']
    # Add both together 
    cards = cards + face_cards
    #print(cards)

    # adding their symbols 

    def prepend(list, str):
        str += '{0}'
        list = [str.format(i) for i in list]
        return(list)

    #define the name of cards
    c_str = 'Club_'
    d_str = 'Diamond_'
    h_str = 'Heart_'
    s_str = 'Spade_'

    # print(prepend(cards, c_str))
    clubs = prepend(cards, c_str)
    diamonds = prepend(cards, d_str)
    hearts = prepend(cards, h_str)
    spades = prepend(cards, s_str)
  
    # combining all 4 diff cards together 

    deck = clubs + diamonds + hearts + spades 

    #print(deck)
    return(deck)


deck = play_cards(10)

# Shuffle Deck #

rand_deck = random.sample(deck, len(deck))
# print(rand_deck)

# Split cards to 4 seperate piles #

chunk_size = len(rand_deck) // 4

split_deck = [rand_deck[i:i + chunk_size] for i in range(0, len(rand_deck), chunk_size)]

# print(split_deck[0])

# labeling my chucks  #

labeled_chucnks = {
    "Players_Hand": split_deck[0],
    "AI1": split_deck[1],
    "AI2": split_deck[2],
    "AI3": split_deck[3],

}

Players_Hand = labeled_chucnks["Players_Hand"]
AI1_h = labeled_chucnks["AI1"]
AI2_h = labeled_chucnks["AI2"]
AI3_h = labeled_chucnks["AI3"]

#print(Players_Hand)

 # Creat a stopwatch #

def countdown(h, m, t):
# Calc total number of seconds 

    total_seconds = h * 3600 + m * 60 + t

# Check if it reaches zero 
# IF not zero, subtract 1 to time
    while total_seconds > 0:

        # Time left on countdown

        timer = datetime.timedelta(seconds = total_seconds)
        
        # Prints the time left on the timer
        print(timer, end="\r")
        
        # Delays the program one sec

        time.sleep(1)

        # Subtract 1 

        total_seconds -= 1

    preform_action()

def preform_action(): 
    s = 0
    return s 

# countdown(int(h), int(m), int(s))
sec = [1,2,3,4,5]

# Makes the Played pile a list 
Played_pile = []

Card_Played_Title = fontstyle.apply("_Card Played_",'bold')

# Creates the timed input 
def timed_input(): 
    push_it = input("Slap!!: ")

# Playing the game 

def Ai_play_cards1():

    print("_____Opponent:AI_1_____")
    print(f'AI_1 cards: {len(AI1_h)}')
    print("__________________________")
    print(" ") # space 
    # Cards played


    print(Card_Played_Title)
    print("  ")
    time.sleep(3)
    played_card1 = AI1_h.pop(0)
    print(played_card1)

    print(' ')
    time.sleep(3)
    # Stores Played Cards 

    Played_pile.append(played_card1)

   
    print(" ")

    #Tracks if certain card is played 
    if 'Jack' in played_card1:


        if first_slap() == True:
            Slap_player()
        else:
            chose_choice = random.choice(Lists_choices)() # chooses between Functions
    elif Marriage() == True:
        if first_slap() == True:
            Slap_player()
        else:
            chose_choice = random.choice(Lists_choices)() # chooses between Functions

            
    elif doubles() == True: 
        Slap_player

        if first_slap() == True:
                Slap_player()
        else:
            chose_choice = random.choice(Lists_choices)() # chooses between Functions
    else: 
        print("Played Pile: ", end='')
        print(len(Played_pile))
        time.sleep(5)
        print(" ")

    
    

    """"""
def Ai_play_cards2():
    print("_____Opponent:AI_2_____")
    print(f'AI_2 cards: {len(AI2_h)}')
    print("__________________________")
    print(" ") # space 
    # Cards played

    print(Card_Played_Title)
    print("  ")
    time.sleep(3)
    played_card2 = AI2_h.pop(0)
    print(played_card2)

    print(' ')


    # Stores Played Cards 
    time.sleep(3)
    Played_pile.append(played_card2)
    
    print(" ")

    #Tracks if certain card is played 
    if 'Jack' in played_card2:

       
        if first_slap() == True:
            Slap_player()
        else:
            chose_choice = random.choice(Lists_choices)() # chooses between Functions

    elif Marriage() == True:
        if first_slap() == True:
            Slap_player()
        else:
            chose_choice = random.choice(Lists_choices)() # chooses between Functions

            
    elif doubles() == True: 
        Slap_player

        if first_slap() == True:
                Slap_player()
        else:
            chose_choice = random.choice(Lists_choices)() # chooses between Functions
    else: 
        print("Played Pile: ", end='')
        print(len(Played_pile))
        time.sleep(5)
        print(" ")
    

    """"""
def Ai_play_cards3():
    print("_____Opponent:AI_3_____")
    print(f'AI_3 cards: {len(AI3_h)}')
    print("__________________________")
    print(" ") # space 
    # Cards played

    print(Card_Played_Title)
    print("  ")
    time.sleep(3)
    played_card3 = AI3_h.pop(0)
    print(played_card3)

    # Stores Played Cards 
    time.sleep(3)
    Played_pile.append(played_card3)
    print(" ")

#Tracks if certain card is played 
    if 'Jack' in played_card3:


        if first_slap() == True:
            Slap_player()
        else:
            chose_choice = random.choice(Lists_choices)() # chooses between Functions
            
    elif Marriage() == True:
        if first_slap() == True:
            Slap_player()
        else:
            chose_choice = random.choice(Lists_choices)() # chooses between Functions

            
    elif doubles() == True: 
        Slap_player

        if first_slap() == True:
                Slap_player()
        else:
            chose_choice = random.choice(Lists_choices)() # chooses between Functions

    else: 
        print("Played Pile: ", end='')
        print(len(Played_pile))
        time.sleep(5)
        print(" ")

    ''''''''''''''
def Player_play_cards():
    print("____You____")
    print(f'Your cards: {len(Players_Hand)}')
    print("__________________________")
    print(' ')
    
    # Play a card from deck #
   
# Play a card from deck (delete a card from their hand, but store them in new list)
    print(Card_Played_Title)
    print("  ")
    time.sleep(3)

    played_card = Players_Hand.pop(0)
    print(played_card)
    time.sleep(3)
#Tracks if certain card is played 
    if 'Jack' in played_card:

        
        if first_slap() == True:
            Slap_player()
        else:
            chose_choice = random.choice(Lists_choices)() # chooses between Functions

    elif Marriage() == True:
        if first_slap() == True:
            Slap_player()
        else:
            chose_choice = random.choice(Lists_choices)() # chooses between Functions

            
    elif doubles() == True: 
        Slap_player

        if first_slap() == True:
                Slap_player()
        else:
            chose_choice = random.choice(Lists_choices)() # chooses between Functions        
    else: 
        print(' ')

        # Stores Played Cards 
        print("Played Pile: ", end='')

        Played_pile.append(played_card)

        print(len(Played_pile))
        
        print(" ")

# The Jack card was played 
def Slap_A1():

    print("Oppenent AI_1 Slap it first")
    AI1_h.extend(Played_pile) # Need to use extend in order to make one list 
    random.shuffle(AI1_h) # Shuffles new deck
        
    if preform_action() == 0:

        Played_pile.clear()
        print(" ")
        print(len(Played_pile))   
def Slap_A2():
    print("Oppenent AI_2 Slap it first")
    AI2_h.extend(Played_pile) # Need to use extend in order to make one list
    random.shuffle(AI2_h) # shuffles new deck
        
    if preform_action() == 0:

        Played_pile.clear()
        print(" ")
        print(len(Played_pile))   
def Slap_A3():
    print("Oppenent AI_3 Slap it first")
    AI3_h.extend(Played_pile) # Need to use extend in order to make one list 
    random.shuffle(AI3_h) # shuffles new deck

    if preform_action() == 0:
            
        Played_pile.clear()
        print(" ")
        print(len(Played_pile))
def Slap_player():
    print("You got it!!")
    Players_Hand.extend(Played_pile)  # Need to use extend in order to make one list 
    random.shuffle(Players_Hand) # shuffles new hand  

    if preform_action() == 0:
            
        Played_pile.clear()
        print(" ")
        print(len(Played_pile))

# Slapping for player 
def first_slap():
    # Create a thread to run push_it
    input_thread = threading.Thread(target=timed_input)

    #Start the thread 
    input_thread.start()
    input_thread.join(timeout=t)

    if input_thread.is_alive():
        input_thread.join()
        print("Time's up!")
        return False
    else:
        return True
    
    
        


def Marriage():

    if len(Played_pile) > 1:
        second_card = Played_pile[-2]

        last_card = Played_pile[-1]

        split_it = last_card.split('_')

        split_it2 = second_card.split('_')

        # Checks if last two cards are a Queen or King 
        if(split_it2[-1] in ["Queen", "King"]) and (split_it[-1] in ["Queen", "King"]):
            return True
    else:
        pass

def doubles():
    if len(Played_pile) > 1:
        second_card = Played_pile[-2]

        last_card = Played_pile[-1]

        split_it = last_card.split('_')

        split_it2 = second_card.split('_')

    
        # Checks if last two cards are doubles
        if(split_it2[-1] == split_it[-1]):
            return True
    else:
        pass

Lists_choices = [Slap_A1,Slap_A2,Slap_A3]
# Print out Player Amount of Cards
# Split the last two cards 

print("Would you like to start the game? Press Enter")
start_game = input("Click x if you don't want to play and Enter if you do: ")
while True: 
    if start_game.lower() == 'x':
        False
    else:  
        x = 1
        while x > 0: 
            t = random.choice(sec)
            count_less_than_zero = 0
            print(" ") # space
            if len(AI1_h) > 0:
                Ai_play_cards1()
            else: 
                count_less_than_zero += 1
                pass
            if len(AI2_h) > 0:
                Ai_play_cards2()
            else:
                count_less_than_zero += 1
                pass
            if len(AI3_h) > 0: 
                Ai_play_cards3()
            else:
                count_less_than_zero += 1
                pass
            if count_less_than_zero >= 3:
                print("Game over!! You won!!")
                break
            else: 
                pass
            if len(Players_Hand) > 0:
                Player_play_cards()
                # Option to play or timer

                '''
                print("Press Enter to continue")
                play_q = input("End the Game Early? Press X: ")

                if play_q.lower() == 'x':
                    x -= 1
                else:
                    x = 1
                '''
            else: 
                print("Game over!!")
                print("You won!!")
                break

            






