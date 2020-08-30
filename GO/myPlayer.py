# -*- coding: utf-8 -*-

import time
import Goban 
import math
from random import randrange
from random import randint
from random import choice
from random import shuffle
from playerInterface import *
import datetime
import json
import os
import numpy as np

# Use model prediction
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json

# Little class representing board structure required by the NN
# Since the role of this class isnt implementing a whole new board, it uses Goban.Board()
# Especially board.next_player() to assign a color to a move
# That's why it is mandatory to make any push() (respectively pop()) before a Goban.Board.push() (respectively pop())
class NNboard():
    
    def __init__(self, board):
        
        self._board = board
        self._NNboard = np.zeros([1, board._BOARDSIZE, board._BOARDSIZE, 2])
        
    # push move (col, lin) in the board (i.e set position value to 1)
    # color can be 0 for black or 1 for white
    def push(self, move):
        self._NNboard[0][move[0]][move[1]][self._board.next_player() - 1] = 1
        
    # pop move (col, lin) in the board (i.e set position value to 0)
    # color can be 0 for black or 1 for white 
    def pop(self, move):
        self._NNboard[0][move[0]][move[1]][1 - (self._board.next_player() - 1)] = 0
    
    def getBoard(self):
        return self._NNboard
    
    def __str__(self):
        return self._NNboard.__str__()
        
               
class myPlayer(PlayerInterface):
   

    def __init__(self):
        self._board = Goban.Board()
        self._NNboard = NNboard(self._board)
        self._mycolor = None
        self._turn = 0
        
        # Starting depth
        self._depth = 2

        # Importants steps in the game (for MinMAx and AlphaBeta strategies)
        self._mid_game = 20
        self._late_game = 40
        self._very_late_game = 50
        
        # Maximal time for a move (for Iterative Deepening)
        self._timeout = 5
        # T0 of a move
        self._t0 = None
        
        # List storing time taken by the player at each move
        self._time_history = []
        
        # Openings
        self._opening_length = 5  # Number of opening moves the player will play
        self._opening = None      # Will contain an opening
        self._opening_index = 0   # To keep track of where to start checking opening moves
        self._openings = []
        
        # Model
        self._model = None
        
        if os.path.exists('games.json'):
            with open('games.json') as json_file:
                data = json.load(json_file)
            for g in data:
                self._openings.append(g['moves'])
        else:
            self._openings = [
            ["C6", "F4", "D4", "F6", "H5", "D3", "C3", "D5", "C4", "E4", "G7", "F7", "G8", "H4", "G5", "G4", "D6", "F8", "G6", "D2", "C2", "D8", "C8", "E5", "J4", "J3", "J5", "C5", "B5", "B7", "B6", "C7", "H3", "A6", "D7", "B8", "E8", "C9", "F2", "G3", "H2", "G2", "F9", "E2", "E9", "E7", "G9", "D9", "F3", "H1", "J2", "C1", "D1", "B1", "B2", "E1", "A4", "A7"],
            ["E5", "E7", "E3", "E2", "D2", "G6", "D7", "D8", "D6", "C8", "G5", "H5", "H6", "G4", "G7", "F5", "F6", "E4", "G5", "F3", "D3", "G6", "E8", "D4", "C4", "D5", "E6", "C5", "B4", "G5", "F2", "G2", "C6", "B5", "B6", "A6", "A7", "A5", "B8", "B2", "C2", "F1"],
            ["E5", "E7", "E3", "E2", "D2", "G6", "G5", "H5", "D7", "D8", "C7", "E6", "F5", "D6", "F6", "G7", "H4", "C6", "B5", "D3", "D4", "C3", "C2", "C4", "C5", "D5", "E4", "B6", "B3", "A5", "B4", "H6", "F7", "F8", "J4", "J5", "A4", "A6"],
            ["E5", "E7", "E3", "E2", "D2", "F3", "E4", "G5", "G4", "G3", "D7", "D8", "C7", "C8", "F7", "E6", "F6", "F8", "D6", "E8", "B8", "H4", "F5", "H7", "H5", "F4", "H6", "H8", "G6", "E1", "H2", "C2", "D3", "D1", "G2", "F2", "H3", "J4", "C3", "G4", "B2", "G1", "H1", "J2", "C1", "J3", "F1", "B7", "B6", "G1", "G7", "G8", "F1", "B9", "J5", "G1", "C9", "J1", "A9", "D9", "J7", "B9", "A7", "A8", "B7", "J8", "A9", "C9", "A8", "J6", "PASS", "J7"], 
            ["E5", "G5", "G4", "C6", "C5", "D5", "D4", "D6", "C4", "H4", "F4", "E6", "H5", "F5", "H6", "E4", "E3", "H3", "G7", "G2", "E8", "D8", "E7", "E2", "D2", "F2", "D7", "C7", "E5", "B5", "B4", "E4", "F3", "B8", "B6", "B7", "A5", "E9", "H8", "F9", "G8", "D1", "E1", "F1", "C2", "J4", "C9", "D3", "E5", "D9", "B9", "E4", "C3", "G3", "E5", "A7", "A9", "E4", "D3", "A6", "B5", "B2", "B1", "A3", "C1", "A1", "E5"], 
            ["E4", "D6", "E6", "E7", "F6", "D4", "D3", "D5", "E3", "F7", "G7", "G8", "D7", "G6", "H7", "D8", "C7", "C8", "B6", "B7", "C6", "B5", "A6", "B8", "B4", "C4", "B3", "G5", "F4", "G4", "G3", "H3", "H2", "G2", "F2", "F3", "F8", "E8", "G3", "A5"], 
            ["G4", "D5", "D3", "E7", "G7", "F5", "G5", "E3", "D4", "E4", "C5", "C6", "D6", "E5", "B6", "C7", "B4", "D2", "B7", "G8", "H8", "F7", "H6", "C3", "C4", "C8", "C2", "F2", "D1", "E1", "E2", "H7", "J7", "D2", "G2", "G3", "H3", "B2", "E2", "H2", "F1", "G1", "F8", "E1", "D2", "E8", "G9", "H4", "B8", "B9", "H5", "J3", "F4", "F3", "J5", "E9", "F9", "J4", "F6", "E6", "A8", "F1", "G6"], 
            ["G4", "C6", "E7", "E5", "E3", "G5", "G7", "F4", "F3", "H7", "H4", "G6", "C7", "D6", "D7", "B7", "B8", "B5", "A7", "A6", "B6", "G2", "G3", "B7", "F6", "F5", "B6", "D2", "A5", "A4", "D3", "C2", "C3", "B3", "C5", "A6", "D5", "E6", "A5"], 
            ["E5", "E7", "F7", "F3", "E6", "D7", "F8", "C6", "C5", "B5", "B4", "C4", "D5", "B6", "C3", "G5", "C8", "D4", "E3", "B3", "E4", "A4", "G3", "G6", "H7", "G7", "G8", "H8", "H9", "G2", "G4", "F4", "F2", "F5", "H2", "E2", "G1", "D3", "D2", "C2", "E8", "D1"], 
            ["G4", "D5", "D3", "G6", "D7", "F7", "B7", "C3", "C2", "D2", "E2", "D4", "E3", "B2", "D1", "B4", "E8", "H4", "H5", "G5", "H3", "F4", "G3", "F8", "F9", "B6", "G8", "B8", "H6", "G7", "H7", "H8", "G9", "C7", "C8", "A7", "C6", "D6", "B7", "H9", "E9", "C7", "E7", "E6", "B7"], 
            ["C5", "G5", "F5", "G6", "F4", "G4", "F6", "F7", "E7", "F3", "G7", "F8", "E8", "E3", "H7", "D5", "D4", "C4", "D6", "D3", "E5", "B4", "H6", "H5"], 
            ["E5", "E7", "E3", "E2", "F2", "C6", "C5", "B5", "D6", "C4", "D5", "C7", "D7", "F3", "F4", "G3", "D2", "G2", "E1", "D8", "H5", "F8", "B3", "B4", "G7", "G8", "H8", "F6", "G6", "H9", "J8", "C3", "B2", "C2", "C1", "G4", "F5", "G5", "F7", "H4", "J6", "E8", "E6", "H2", "J3", "J4", "G1", "H1", "J2", "F1", "C8", "E2", "B8", "B6", "F2", "B1", "E2", "D1", "D9", "E4", "D4", "D3", "C1", "H6", "A1", "J5", "H7", "J7", "E9", "C9", "B9", "G9", "C9"], 
            ["E5", "E3", "F3", "F4", "E4", "G3", "F2", "D3", "C4", "G2", "G4", "F5", "F6", "G5", "G6", "H4", "H5", "G4", "C6", "C3", "B4", "C7", "B7", "D7", "B8", "H7", "H6", "F8", "H8", "G8", "G7", "H9", "J8", "E6", "E7", "D6", "D5", "D8", "F7", "E8", "G9", "J5", "J7", "J4", "J6", "B6", "C5", "C8", "C9", "B9", "A9", "D9", "F9", "B9", "A6", "A8", "B5", "B3", "A4", "E9", "G9"], 
            ["E6", "D4", "F4", "C6", "D3", "G7", "C4", "D5", "E3", "C3", "B3", "F5", "F6", "C2", "B5", "B6", "B2", "E4", "G4", "F3", "G3", "F2", "D2", "G5", "G6", "H5", "H6", "G2", "D7", "J6", "J7", "J5", "H7", "C8"], 
            ["E6", "E4", "D4", "D3", "D5", "C3", "F4", "F3", "G4", "G3", "E7", "B5", "B4", "C4", "B6", "C5", "C6", "E5", "F5", "D6", "D7", "H4", "H5", "F6", "G5", "H6"], 
            ["E5", "E7", "C6", "E3", "G4", "G6", "F7", "F6", "E6", "F8", "D7", "G7", "C3", "C2", "B2", "D3", "C4", "F4", "G2", "F5", "H5", "D8", "C8", "C9", "B9", "D9", "B8", "F2", "G3", "H6", "C1", "D2", "J2", "B1", "A1", "D1", "B3", "B1", "A2", "G1", "H1", "F1", "J6", "J7", "J5", "G5", "H4", "D4", "D5", "C1"], 
            ["G4", "D5", "D3", "F6", "C7", "C5", "F8", "C3", "F3", "D2", "E2", "D4", "E3", "G5", "E7", "G8", "B6", "F7", "E8", "H5", "H4", "J4", "J3", "J5", "H2", "B5", "G9", "H8", "C2", "B2", "D1", "B8", "B7", "C8", "A8", "B1", "B3", "C4", "H9", "F9", "E9", "A6", "D8", "B9", "H7", "G7", "E6", "D6", "E5", "E4", "F5", "H6", "D7", "F4", "A2"]
        ]
        
        # Retrieve learnt model saved on disk (for heuristic)
        # load json and create model
        json_file = open('./model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self._model = tf.keras.models.model_from_json(loaded_model_json)
        # load weights into new model
        self._model.load_weights("./model.h5")

        # evaluate loaded model on test data
        self._model.compile(loss='mae', optimizer='adam', metrics=['accuracy'])
                     
    def getPlayerName(self):
        return "IAplusdepapierdanslesWC"

    def getPlayerMove(self):
        if self._board.is_game_over():
            print("Referee told me to play but the game is over!")
            return "PASS"
        
        # TODO for any strategy
        if(self._board._lastPlayerHasPassed) :
            score = self._board.compute_score()
            if(score[self._mycolor-1] > score[2-self._mycolor]) :
                self._turn += 1 # Not really important but to stay consistent (:
                return "PASS"
        
        depth = None # Will change with ID calls, stored to see how deep we go
        
        # Chose a random opening
        if self._turn == 0:
            print("Playing {} opening moves".format(self._opening_length))
            self._opening = self.get_opening()
         
        # Used to compute time taken to play each move
        startTime = time.time()
        
        # Play self._opening_length opening moves
        if self._turn < self._opening_length:
            (move, val, n, f)  = (self.PlayOpeningMove(), None, None, None)
            self._turn += 1
        
        # Start early game
        else:
            # Change depth according to the current state of the game
            if self._turn >= self._mid_game and self._turn < self._late_game:
                self._depth = 2
            elif self._turn >= self._late_game and self._turn < self._very_late_game:
                self._depth = 3
            elif self._turn >= self._very_late_game:
                self._depth = 4

            
            # MinMax
            #(move,val, n, f) = self.MaxMinMove(self._board, self._depth)
            
            # ALPHA BETA
            #(move,val, n, f) = self.AlphaBeta(self._board, self._depth)
            
            # ID
            self._t0 = time.time()
            (move , depth) = self.IterativeDeepening()
            self._turn += 1
            
        # Used to compute time used to play 1 move
        totalTime = time.time() - startTime
        
        #Store time for this move
        self._time_history.append(totalTime)
               
        if self._model is not None:
            print("Evaluating with model")
            
        #print("Valeur du coup : ", val)
        #print("Nodes: ", n)
        #print("Leaves: ", f)  

        self._NNboard.push(self._board.unflatten(move))
        self._board.push(move)
        
        # New here: allows to consider internal representations of moves
        print("I am playing ", self._board.move_to_str(move))
        print("It took me {} minute(s) and {:02.1f} second(s)".format(int(totalTime//60), totalTime%60))
        if depth is not None:                                                     
            print("Went to depth {}".format(depth))
        print("My current board :")
        self._board.prettyPrint()
        # move is an internal representation. To communicate with the interface I need to change if to a string

        return Goban.Board.flat_to_name(move) 

    def playOpponentMove(self, move):
        print("Opponent played ", move) # New here
        # the board needs an internal represetation to push the move.  Not a string
        self._NNboard.push(Goban.Board.name_to_coord(move))
        self._board.push(Goban.Board.name_to_flat(move)) 

    def newGame(self, color):
        self._mycolor = color
        self._opponent = Goban.Board.flip(color)

    def endGame(self, winner):              
        if self._mycolor == winner:
            print("Je suis beaucoup trop fort en fait!!!")
        else:
            print("Tu as eu de la chance!!")
        print(self._time_history)
            
    ########################## Opening strategies ##########################################################################

    # Return a flat opening move (legal and superKO checked)
    def PlayOpeningMove(self):
        
        if self._opening == None:
            print("Opening not found...")
            return -1 #PASS
        
        if self._opening_index <= len(self._opening):
            
            for k in range(self._opening_index, len(self._opening)):

                move = self._board.name_to_flat(self._opening[k])

                # Check if move is playable
                if move in self._board.weak_legal_moves():

                    # Update NN board first
                    self._NNboard.push(self._board.unflatten(move))
                    # SuperKO check
                    if self._board.push(move):

                        self._NNboard.pop(self._board.unflatten(move))
                        self._board.pop()
                        self._opening_index = k + 1 
                        return move
                    
                    self._NNboard.pop(self._board.unflatten(move))
                    self._board.pop()
        
        # Case no appropriate move was found or reached the end of chosen opening
        self._opening = self.get_opening()
        self._opening_index = 0
        return self.PlayOpeningMove()
               
    #Creates the opening moves list from a random player and random opening in self._openings
    def get_opening(self):
        
        lib_length = len(self._openings)
        
        #Select a random game in the library
        chosen_opening = self._openings[randrange(0, lib_length)]
        
        #Pick a random player
        player = randint(0, 1)
        
        #Build opening list
        opening = [chosen_opening[k] for k in range(player, len(chosen_opening) ,2)]
        
        print("Chosen player is {} and the opening is {}".format(player, opening))
        
        return opening
    
    # Costly way to reshape board to pass it to neural network
    # NNboard replaced it
    def reshape_board(self, b):
        
        reshaped_board = np.zeros([1, b._BOARDSIZE, b._BOARDSIZE, 2], dtype = int)
        
        for k in range(b._BOARDSIZE**2):
            
            # Get coordinates on board
            (col, lin) = b.unflatten(k)
            
            if b[k] == 1: #Case black
                reshaped_board[0][col][lin][0] = 1
            elif b[k] == 2: #Case white
                reshaped_board[0][col][lin][1] = 1
            
        return reshaped_board      
    
    
    # Heuristic of the board
    def evaluate(self, b):

        # Use learnt heuristic
        if self._model is not None:
            pred = self._model.predict(self._NNboard.getBoard())[0][self._mycolor - 1] * 100
            return pred
            
        # In case there is an error loading model
        else:
            if self._mycolor == Goban.Board._WHITE:
                v = (b._nbWHITE * 1. / (b._BOARDSIZE**2 - len(b._empties))) * 100
                #v += (whites*2)-blacks
            else:
                v = (b._nbBLACK * 1. / (b._BOARDSIZE**2 - len(b._empties))) * 100
                #v += (blacks*2)-whites
            return v#+others
        
       
    ########################## Strategies ##########################################################################
    
    ################## MinMax #########################################################################################
    
    def MaxMinMove(self, b, depth):
        # If depth is zero : PASS
        if depth == 0:
            return (-1, -1, 0, 1)
        
        (coup, v) = None, None
        
        #Get legal moves (without superKO check) and shuffle them to be less deterministic
        legal_moves = b.weak_legal_moves()
        shuffle(legal_moves)
        
        # Loop to get the best move
        for move in legal_moves:
            
            self._NNboard.push(self._board.unflatten(move))
            if b.push(move): #Continue only if move is not a superKO
                (ret, n, f) = self.MinMax(b, depth - 1)
            
                # If the move we checked is better we take it
                if v is None or ret > v:
                    coup = move
                    v = ret
            
            self._NNboard.pop(self._board.unflatten(move))
            b.pop() #Pop the move (superKO or not)
        
        # Return the best move and its value
        return (coup,v,n,f)
                
    def MinMax(self, b, depth = 3):
        # Si la partie est finie
        if b.is_game_over():
            res = b.result()
            # If we win (1-0 if white) we want to play this move (huge eval value)
            if res == "1-0":
                return (1000, 0, 1) if (self._mycolor == Goban.Board._WHITE) else (-1000, 0, 1)
            # Same for blacks
            if res == "0-1":
                return (-1000, 0, 1) if (self._mycolor == Goban.Board._WHITE) else (1000, 0, 1)
            else:
                return (0, 0, 1)
        
        # Horizon
        if depth == 0:
            return (self.evaluate(b), 0, 1)
        
        v = None
        (n, f) = (0, 0)
        
        #Get legal moves (without superKO check) and shuffle then to be less deterministic
        legal_moves = b.weak_legal_moves()
        shuffle(legal_moves)
        
        for move in legal_moves:
            n += 1
            
            self._NNboard.push(self._board.unflatten(move))
            if b.push(move): #Case it is not a superKO
                (ret, nn, nf) = self.MaxMin(b, depth - 1)
                if v is None or ret < v:
                    coup = move
                    v = ret
                n += nn
                f += nf
                
            self._NNboard.pop(self._board.unflatten(move))
            b.pop() #Pop in any case
        
        return (v, n, f)
    
    def MaxMin(self, b, depth = 3):
        # Case game is over
        if b.is_game_over():
            res = b.result()
            # If we win (1-0 if white) we want to play this move (huge eval value)
            if res == "1-0":
                return (1000, 0, 1) if (self._mycolor == Goban.Board._WHITE) else (-1000, 0, 1) 
            # If we lose it's the opposite
            if res == "0-1":
                return (-1000, 0, 1) if (self._mycolor == Goban.Board._WHITE) else (1000, 0, 1)
            else:
                return (0, 0, 1)
            
        # Horizon
        if depth == 0:
            return (self.evaluate(b), 0, 1)
        
        v = None
        (n, f) = (0, 0)
        
        #Get legal moves (without superKO check) and shuffle then to be less deterministic
        legal_moves = b.weak_legal_moves()
        shuffle(legal_moves)
        
        for move in legal_moves:
            n += 1
            
            self._NNboard.push(self._board.unflatten(move))
            if b.push(move): #Case it is not a superKO
                (ret, nn, nf) = self.MinMax(b, depth - 1)
                if v is None or ret > v:
                    coup = move
                    v = ret   
                n += nn
                f += nf
                
            self._NNboard.push(self._board.unflatten(move))
            b.pop() #Pop in any case
        
        return (v, n, f)
        
        
    ########################## AlphaBeta ###################################################################################
    
    def AlphaBeta(self, depth = 3, ID = False):
        
        if depth <= 0:
            return (-1, -1, 0, 1)

        (moves, val_max) = [], None
                
        #Get legal moves (without superKO check) 
        legal_moves = self._board.weak_legal_moves()
                      
        (n, f) = (0, 0)
        
        for m in legal_moves:
            
            # Case the algorithm is used by Iterative Deepening
            # Need to stop it if we timeout
            if ID:
                if time.time() - self._t0 > 0.9 * self._timeout:
                    break
                    
            n += 1
            
            self._NNboard.push(self._board.unflatten(m))
            if self._board.push(m):
                
                if ID: # Keep specifying that it is called by ID
                    (ret, nn, nf) = self.MinMaxAlphaBeta(depth-1, -math.inf, math.inf, True)
                else:
                    (ret, nn, nf) = self.MinMaxAlphaBeta(depth-1, -math.inf, math.inf)

                n += nn
                f += nf
                    
                if val_max is None or ret > val_max:
                    moves = []
                    moves.append(m)
                    val_max = ret
                    
                # If the move as max_value add it in playable moves
                elif ret == val_max:
                    moves.append(m)
                    
            self._NNboard.pop(self._board.unflatten(m))        
            self._board.pop()
            
        if len(moves) == 0:
            return (-1, val_max, n, f)

        return (choice(moves), val_max, n, f) 
     
    def MaxMinAlphaBeta(self, depth, alpha, beta, ID = False):
        maxi = alpha
              
        if self._board.is_game_over():
            res = self._board.result()
            if res == "1-0":
                return (1000, 0, 1) if (self._mycolor == Goban.Board._WHITE) else (-1000, 0, 1)
            if res == "0-1":
                return (-1000, 0, 1) if (self._mycolor == Goban.Board._WHITE) else (1000, 0, 1)
            else:
                return (0,0,1)
            
        if(depth <= 0) :
            return (self.evaluate(self._board), 0, 1)
        
        (n, f) = (0, 0)

        for m in self._board.weak_legal_moves():
            
            # Case the algorithm is used by Iterative Deepening
            # Need to stop it if we timeout
            if ID:
                if time.time() - self._t0 > 0.9 * self._timeout:
                    break
                    
            n += 1
            
            self._NNboard.push(self._board.unflatten(m))
            if self._board.push(m):
                
                if ID:
                    (ret, nn, nf) = self.MinMaxAlphaBeta(depth-1, maxi, beta, True)
                else:
                    (ret, nn, nf) = self.MinMaxAlphaBeta(depth-1, maxi, beta)
                    
                n += nn
                f += nf

                maxi = max(maxi, ret)
            
            self._NNboard.pop(self._board.unflatten(m))
            self._board.pop()

            if(maxi >= beta): 
                return (beta, n, f)
            
        return (maxi, n, f)
              
    def MinMaxAlphaBeta(self, depth, alpha, beta, ID = False):
        mini = beta
        
        if self._board.is_game_over():
            res = self._board.result()
            if res == "1-0":
                return (1000, 0, 1) if (self._mycolor == Goban.Board._WHITE) else (-1000, 0, 1)
            if res == "0-1":
                return (-1000, 0, 1) if (self._mycolor == Goban.Board._WHITE) else (1000, 0, 1)
            else:
                return (0,0,1)
            
        if(depth <= 0) :
            return (self.evaluate(self._board), 0, 1)

        (n, f) = (0, 0)
        
        for m in self._board.weak_legal_moves():
            
            # Case the algorithm is used by Iterative Deepening
            # Need to stop it if we timeout
            if ID:
                if time.time() - self._t0 > 0.9 * self._timeout:
                    break
                    
            n += 1
            
            self._NNboard.push(self._board.unflatten(m))
            if self._board.push(m):
                
                if ID:
                    (ret, nn, nf) = self.MaxMinAlphaBeta(depth-1, alpha, mini, True)
                else:
                    (ret, nn, nf) = self.MaxMinAlphaBeta(depth-1, alpha, mini)

                n += nn
                f += nf

                mini = min(mini, ret)

            self._NNboard.pop(self._board.unflatten(m))
            self._board.pop()

            if(alpha >= mini): 
                return (alpha, n, f)
            
        return (mini, n, f)
              
    ################## Iterative deepening ###################################################################################
        
    def IterativeDeepening(self):

        moves = []
        depth = 1
        self._t0 = time.time()

        # While there's time remaining
        while (time.time() - self._t0) < (0.9 * self._timeout):

            # Find best move we can get with current depth
            moves.append(self.AlphaBeta(depth, True)[:2])
            depth += 1

        move = None
        max_value = -math.inf
        
        # Find the best move among those we got during the allowed time
        for m in moves:
            
            if m[1] is None:
                return (-1, depth)
            
            elif m[1] > max_value:
                move = m[0]
                max_value = m[1]
                         
        return (move, depth - 1)
                
                
                         

            
                             
            
                             
                   
                   
