"""
Completely rebuilt this:
https://github.com/Narengowda/TechDummies/blob/master/boggled_text.py

Enhancements:
- create random boards for testing
- cells can contain multiple letters (e.g. "Qu")
- blank cells that can represent any letter
- dead cells that are not a letter
"""

import time
import string
import random

blank_chr = "_"
dead_chr = "."
lowercase_chrs = list(string.ascii_lowercase)
possible_chrs = lowercase_chrs+["in", "er", "an", "he", "th", "qu"]

# dx, dy, arrow
cell_directions = [
    [-1,-1, "↖"],
    [-1, 0, "←"],
    [-1, 1, "↙"],
    [0, -1, "↑"],
    [0,  1, "↓"],
    [1, -1, "↗"],
    [1,  0, "→"],
    [1,  1, "↘"],
]

def random_letter():
    return random.choice(possible_chrs)

def all_squares(size):
    out = []
    [out.extend([(i,j) for i in range(size[1])]) for j in range(size[0])]
    return out

def load_words():
    # filename = 'corncob_lowercase.txt'
    filename = 'words_alpha.txt'
    with open(filename) as word_file:
        valid_words = set(word_file.read().split())
    # from nltk.corpus import words
    # valid_words = words.words()
    return valid_words

def make_trie(words):
    root = dict()
    for word in words:
        curr = root
        for letter in word:
            curr = curr.setdefault(letter, {})
        curr["is_word"] = True
    return root


class Board():
    def __init__(self, board):
        self.board = board
        self.neighbors = None
        self.min_word_length = 0
        self.trie = self.build_trie(load_words())

    def __str__(self):
        out = "\n"
        for row in self.board:
            for l in row:
                out += f"{l : ^4}"
            out += "\n"
        return out

    @property
    def size(self):
        return (len(self.board), len(self.board[0]))

    @classmethod
    def random(self, size=(4,4), blank=0, dead=0):
        """ size: tuple, (height/rows, width/cols)
            blank: int, replaces this many squares with blanks
            dead: int, replaces this many sqaures with dead
        """
        board = [[random_letter() for i in range(size[1])] for j in range(size[0])]
        # Now replace some squares with blank or dead
        possible_squares = all_squares(size)
        random.shuffle(possible_squares)
        for i in range(blank):
            x, y = possible_squares.pop()
            board[y][x] = blank_chr
        for i in range(dead):
            x, y = possible_squares.pop()
            board[y][x] = dead_chr
        return self(board)

    def build_trie(self, vocab):
        return make_trie(vocab)

    def is_cell_dead(self, cell):
        char = self.board[cell[1]][cell[0]]
        return char==dead_chr

    def is_cell_valid(self, cell):
        x, y = cell
        # Check if it's on the board
        if (x<0) or (y<0) or (x>=self.size[1]) or (y>=self.size[0]):
            return False
        # Check if it's not dead
        return not self.is_cell_dead(cell)

    def get_letter(self, cell):
        """ Returns only the string in the cell. """
        return self.board[cell[1]][cell[0]]

    def get_letters(self, cell):
        """ Return a list of values for the cell.
            Usually this is a list of one string.
            For a blank, this list has all 26 letters.
        """
        char = self.board[cell[1]][cell[0]]
        if char==blank_chr: return lowercase_chrs
        return [char]

    def cell_neighbors(self, cell):
        """ Make list of all possible next moves.
            The moves do not go off the board and do not go to dead spaces.
        """
        x, y = cell
        if self.neighbors==None:
            # Generate all possible cell neighbors only one time
            # This saves compute for each step in the dfs
            rows, cols = self.size
            # Create the 2d array that stores the neighers for each cell
            self.neighbors = [[[] for i in range(cols)] for j in range(rows)]
            # Check each possible direction for all cells
            for curr_cell in all_squares(self.size):
                cx, cy = curr_cell
                for step in cell_directions:
                    dx, dy, arrow = step
                    test_cell = (cx+dx,cy+dy)  # Make the test cell location
                    # Check if going to the test cell is a valid move
                    if self.is_cell_valid(test_cell):
                        self.neighbors[cy][cx].append( (test_cell, arrow) )
        return self.neighbors[y][x]
        
    def depth_first_search(self, cell, visited, node, word, directions, found):
        """ Single step in the DFS. """
        # If the cell is visited, return
        if cell in visited:
            return
        else:  # otherwise mark this cell as visited
            visited.append(cell)

        # Extend the current word with this cell
        # It is impossible to visit a dead cell, so this aways returns a list
        possible_letters = self.get_letters(cell)

        # For each letter, recursively check all neighboring cells for more words
        for letters in possible_letters:
            new_word = word+letters

            # Check if the new letters continue a word in the trie
            # This is a for loop because cells can have multiple letters
            for letter in letters:
                # Check if the current letter is a key for this node
                if letter in node:
                    node = node[letter]
                    keep_going = True
                else:  # There are no words off this node, leave this cell
                    keep_going = False
                    break

            if not keep_going: continue
            
            # If the node is a word, add it to the output
            if "is_word" in node:
                if len(new_word)>=self.min_word_length:
                    found.append([new_word, directions])
    
            # Check the neighbors
            for new_cell, arrow in self.cell_neighbors(cell):
                new_directions = directions+arrow
                # Recursion
                # Use visited[:] becuase the splice operator only copies the references, and not the object
                self.depth_first_search(cell=new_cell, visited=visited[:], node=node, word=new_word, directions=new_directions, found=found)
    
    def solve(self, min_word_length=0):
        """ Returns list of all valid words"""
        print("Solving")
        self.min_word_length = min_word_length
        found = []
        # Start in each cell and do depth first search
        for cell in all_squares(self.size):
            # Start with a new visited set for each starting cell
            if not self.is_cell_dead(cell):  # Can not start on a dead cell
                directions = f"from {self.get_letter(cell)} ({cell[0]+1}, {cell[1]+1}) go "
                self.depth_first_search(cell=cell, visited=list(), node=self.trie, word="", directions=directions, found=found)
        return found


if __name__ == "__main__":
 
    size = (10,10)
    blank = 0
    dead = 0
    board = Board.random(size, blank, dead)

    # board = [
    #     ["e", "t", "s"],
    #     ["a", "e", "s"],
    #     ["e", "y", "e"],
    # ]
    # board = Board(board)

    print(f"Random board: {board}")
    print(f"{board.size=}")
   
    t0 = time.time()
    found = board.solve(min_word_length=4)
    dt = time.time()-t0
    print(f"Found {len(found)} words {dt*1000:.4f} ms")
    
    found_dict = dict()
    for word, direction in found:
        if word not in found_dict:
            found_dict[word] = []
        found_dict[word].append(direction)

    print(f"Found {len(found_dict)} unique words")

    output_str = ""
    with open(f'found.txt', "w", encoding='utf-8') as f:
        for idx, (word, direction) in enumerate(found):
            output_str = f"{idx+1}, {word}, {direction}\n"
            f.write(output_str.encode('utf-8').decode())
