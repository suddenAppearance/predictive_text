import os
import time
import curses

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['PT_LOGGING'] = '0'
import t9
from predictive_text import Model


def draw_menu(stdscr):
    global model
    k = 0
    cursor_x = 0
    cursor_y = 0

    # Clear and refresh the screen for a blank canvas
    stdscr.clear()
    stdscr.refresh()

    # Start colors in curses
    curses.start_color()
    curses.init_pair(1, curses.COLOR_CYAN, curses.COLOR_BLACK)
    curses.init_pair(2, curses.COLOR_RED, curses.COLOR_BLACK)
    curses.init_pair(3, curses.COLOR_BLACK, curses.COLOR_WHITE)
    curses.init_pair(4, curses.COLOR_GREEN, curses.COLOR_BLACK)
    curses.init_pair(5, curses.COLOR_YELLOW, curses.COLOR_BLACK)
    written_text = ""
    suggested_text = ""
    predicted_word = ""
    t9_word = ""
    last_is_space = ""

    # Loop where k is the last character pressed
    while k != 27:
        # Initialization
        stdscr.clear()
        height, width = stdscr.getmaxyx()

        cursor_x = max(0, cursor_x)
        cursor_x = min(width - 1, cursor_x)

        cursor_y = max(0, cursor_y)
        cursor_y = min(height - 1, cursor_y)

        # creating input simulation
        if ord('А') <= k <= ord('Я') or ord('а') <= k <= ord('я') or k == ord(' ') or k == ord('\n'):
            written_text += chr(k)
            last_is_space = k == ' '
        elif k == ord('\b'):
            written_text = written_text[:len(written_text) - 1]
        elif k == ord('\t'):
            if predicted_word != '' and predicted_word is not None:
                written_text += (" " if written_text[len(
                    written_text) - 1] != " " else '') + predicted_word  # predicted word will be rewritten after
            else:
                written_text += t9_word
        abstracts = written_text.split('\n')
        i = 0
        j = 0
        while i < len(abstracts):
            abstract = abstracts[i]
            if len(abstract) < width:
                stdscr.addstr(j, 0, abstract)
                i += 1
                j += 1
                cursor_x = len(abstract) + 1
            else:
                cut_words = abstract[:width].split()
                stdscr.addstr(j, 0, " ".join(cut_words[:-1]))
                cursor_x = len(abstract)
                j += 1
                print(abstract)
                abstracts[i] = cut_words[-1] + abstract[width:]
        cursor_y = j - 1
        predicted_word = model.buildPhrase(written_text.split('\n')[-1])
        if not (predicted_word == '' or predicted_word is None):
            stdscr.attron(curses.color_pair(1))
            if not last_is_space:
                stdscr.addstr(cursor_y, cursor_x - 1, ' ')
            stdscr.attron(curses.color_pair(4))
            stdscr.addstr(predicted_word)
            stdscr.attroff(curses.color_pair(4))
        elif written_text:
            stdscr.attron(curses.color_pair(5))
            try:
                last = written_text.split('\n')[-1].split()[-1]
            except IndexError:
                last = ''
            t9_word = t9.T9().complete(last)
            stdscr.addstr(t9_word)
            stdscr.attroff(curses.color_pair(5))
        stdscr.attroff(curses.color_pair(1))
        # Refresh the screen
        stdscr.refresh()

        # Wait for next input
        k = stdscr.getch()


def main():
    curses.wrapper(draw_menu)


if __name__ == "__main__":
    print('Loading models for predictive text')
    model = Model()
    print('Starting...')
    time.sleep(0.5)
    main()
