import time
import curses
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
    written_text = ""
    suggested_text = ""
    predicted_word = ""

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
        elif k == ord('\b'):
            written_text = written_text[:len(written_text) - 1]
        elif k == ord('\t'):
            written_text += (" " if written_text[len(written_text) - 1] != " " else '') + predicted_word  # predicted word will be rewritten after
        stdscr.attron(curses.color_pair(1))
        rows = written_text.split('\n')
        for i in range(len(rows)):
            stdscr.addstr(i, 0, rows[i])
        cursor_x = len(rows[len(rows) - 1]) + 1
        cursor_y = len(rows)
        predicted_word = model.buildPhrase(rows[len(rows)-1])
        if not (predicted_word == '' or predicted_word is None):
            stdscr.attron(curses.color_pair(1))
            # print(rows)
            # print(len(rows))
            # print(cursor_x)
            # print(cursor_y)
            stdscr.addstr(cursor_y - 1, cursor_x - 1, ' ')
            stdscr.attron(curses.color_pair(4))
            stdscr.addstr(predicted_word)
            stdscr.attroff(curses.color_pair(4))
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
