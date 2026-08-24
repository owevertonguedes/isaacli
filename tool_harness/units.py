"""Numbers as a person reads them, decided once.

A model's size was turned into GiB by hand in ten places across five modules,
in two different precisions, so the same file read as `4.68 GiB` on the screen
that chooses a quantization and `4.7 GiB` in the table right next to it. That
is one quantity with two answers, and changing either one meant finding all
ten.

There are two functions and not one because the two precisions are a real
distinction, not an accident: a number being scanned down a narrow column and
a number being compared against another number are different jobs. What is not
a distinction is which of the two a given call site picked, so the choice is
made here, by name, and never at the call site.

Only GiB, deliberately. A unit that scales itself would print MB for one row
and GiB for the next, and a column whose unit changes per row cannot be
compared down its own length, which is the only reason to put sizes in a
column.

Throughput follows the same split for the same reason, and the pair of names
is deliberately parallel: a quantity that appears both in a cell and in a
sentence gets its two forms here, so that nowhere else has to remember which
is which.
"""

BYTES_PER_GIB = 1024 ** 3


def gib(value):
    """A size in a sentence, or one being compared against another size.

    Two decimals, because this is the form that shows up where the numbers are
    weighed against each other: the quantization screen exists to choose
    between neighbours, and rounding 4.68 and 4.71 to the same 4.7 removes the
    difference the screen was opened for. The fit arithmetic needs it for the
    same reason, so that weights plus cache still visibly adds up to the total.
    """
    return f"{(value or 0) / BYTES_PER_GIB:.2f}"


def gib_short(value):
    """A size in a table cell, where the column is narrow and being scanned.

    One decimal. The suggestion table lands at 79 columns in English and 80 in
    Portuguese, so the second decimal is a column the table does not have; and
    a cell is read against the fit cell beside it, not against the cell above.

    One caller, and that is the point rather than a reason to inline it: every
    screen that draws a model row goes through `model_discovery.model_row`, so
    there is exactly one place a cell is built. Inlining the format there is
    how the ten hand-written copies started.
    """
    return f"{(value or 0) / BYTES_PER_GIB:.1f}"


def tps(value):
    """Tokens per second in a sentence: the evidence line under a choice.

    One decimal, because that line names a measurement, a card and a date, and
    a measurement reported as a round number invites being read as a round
    number somebody chose. The catalog holds 33.02 and 29.37, and the tenth is
    the part of that a person can compare against their own run.
    """
    return f"{value:.1f}"


def tps_short(value):
    """Tokens per second in a table cell: whole tokens, no decimal.

    The column is scanned, not compared to the tenth, and a decimal point in a
    narrow column costs two characters to say nothing anybody acts on. It also
    keeps a measured cell and an estimated cell the same shape, which the
    legend under the table needs: the origin is said once, down there, and a
    cell that carried its own precision would be saying it again per row.
    """
    return f"{value:.0f}"
