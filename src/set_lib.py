import json

from main import Book

from os import path

book_lib = [
    Book(
        title="Advanced Problems in Mathematics",
        authors=["Stephen Siklos"],
        tags=["Maths", "Problem Solving", "Oxbridge"],
        average_rating=5,
        book_id=1
    ),
    Book(
        title="Stanford Mathematics Problem Book",
        authors=["George Polya", "Jeremy Kilpatrick"],
        tags=["Maths", "Problem Solving", "Oxbridge", "Interviews"],
        average_rating=4,
        book_id=2
    ),
    Book(
        title="Professor Povey's Perplexing Problems",
        authors=["Thomas Povey"],
        tags=["Problem Solving", "Physics", "Engineering", "Interviews"],
        average_rating=4,
        book_id=3
    ),
    Book(
        title="Fifty Challenging Problems in Probability",
        authors=["Frederick Mosteller"],
        tags=["Maths", "Problem Solving"],
        average_rating=2,
        book_id=4
    ),
    Book(
        title="Algorithmic Puzzles",
        authors=["Anany Levitin", "Maria Levitin"],
        tags=["Problem Solving", "Computer Science", "Logic", "Interviews"],
        average_rating=4,
        book_id=5
    ),
    Book(
        title="How To Solve It",
        authors=["George Polya"],
        tags=["Maths", "Problem Solving", "Textbook"],
        average_rating=3,
        book_id=6
    ),
    Book(
        title="How To Solve It by Computer",
        authors=["Dromey"],
        tags=["Computer Science", "Logic", "Textbook"],
        average_rating=3,
        book_id=7
    ),
    Book(
        title="Fermat's Last Theorem",
        authors=["Simon Singh"],
        tags=["Maths", "History of Maths", "Biography"],
        average_rating=4.5,
        book_id=8
    ),
    Book(
        title="The Code Book",
        authors=["Simon Singh"],
        tags=["Maths", "History of Maths", "Computer Science"],
        average_rating=4,
        book_id=9
    ),
    Book(
        title="The Great Mathematical Problems",
        authors=["Ian Stewart"],
        tags=["Maths", "History of Maths", "Problem Solving"],
        average_rating=4,
        book_id=10
    ),
    Book(
        title="My Best Mathematical and Logic Puzzles",
        authors=["Martin Gardner"],
        tags=["Maths", "Logic", "Problem Solving"],
        average_rating=3,
        book_id=11
    ),
    Book(
        title="How to Think Like a Mathematician",
        authors=["Kevin Houston"],
        tags=["Maths", "Textbook", "Interviews"],
        average_rating=4.5,
        book_id=12
    ),
]

with open(path.join(path.dirname(__file__), "library.json"), "w") as file:
    data = [
        {
            "title": book.title,
            "authors": book.authors,
            "tags": book.tags,
            "average_rating": book.average_rating,
            "book_id": book.id
        } for book in book_lib
    ]

    json.dump(data, file, indent=4)
