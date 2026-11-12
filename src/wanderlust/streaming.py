"""Word-wrap text that arrives in arbitrary fragments (streamed model output)."""

from __future__ import annotations

from collections.abc import Callable


class WordWrapper:
    """Writes streamed text, breaking lines between words instead of mid-word.

    Fragments can split words anywhere, so each word is held back until the
    whitespace after it arrives. Newlines in the text are kept, and spaces at
    the start of a line are kept too (list indentation).
    """

    def __init__(self, write: Callable[[str], None], width: int) -> None:
        if width < 10:
            raise ValueError("width must be at least 10")
        self._write = write
        self.width = width
        self._column = 0
        self._word = ""
        self._space_pending = False

    def feed(self, text: str) -> None:
        for char in text:
            if char == "\n":
                self._flush_word()
                self._write("\n")
                self._column = 0
                self._space_pending = False
            elif char.isspace():
                if self._column == 0 and not self._word:
                    self._write(" ")  # indentation at the start of a line
                    self._column += 1
                else:
                    self._flush_word()
                    self._space_pending = True
            else:
                self._word += char

    def close(self) -> None:
        """Write any word still held back."""
        self._flush_word()

    def _flush_word(self) -> None:
        if not self._word:
            return
        separator = 1 if self._space_pending else 0
        if self._column > 0 and self._column + separator + len(self._word) > self.width:
            self._write("\n")
            self._column = 0
            separator = 0
        if separator:
            self._write(" ")
            self._column += 1
        self._write(self._word)
        self._column += len(self._word)
        self._word = ""
        self._space_pending = False
