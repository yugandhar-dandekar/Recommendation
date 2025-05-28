import numpy as np

# weights for PageRank
LINK_WEIGHT = 0.85
PREFERENCES_WEIGHT = 0.1
RATINGS_WEIGHT = 1 - (LINK_WEIGHT + PREFERENCES_WEIGHT)

# user preferences
PREFERENCES = ["Engineering", "Physics", "Problem Solving"]


class Book():
    """
    Represents a book in a library
    """
    def __init__(self, title: str, authors: list[str], tags: list[str],
                 average_rating: float, id: int):
        self.title = title
        self.authors = authors
        self.tags = tags
        self.average_rating = average_rating
        self.id = id


# book library data set
book_lib = [
    Book(
        title="Advanced Problems in Mathematics",
        authors=["Stephen Siklos"],
        tags=["Maths", "Problem Solving", "Oxbridge"],
        average_rating=5,
        id=1
    ),
    Book(
        title="Stanford Mathematics Problem Book",
        authors=["George Polya", "Jeremy Kilpatrick"],
        tags=["Maths", "Problem Solving", "Oxbridge", "Interviews"],
        average_rating=4,
        id=2
    ),
    Book(
        title="Professor Povey's Perplexing Problems",
        authors=["Thomas Povey"],
        tags=["Problem Solving", "Physics", "Engineering", "Interviews"],
        average_rating=4,
        id=3
    ),
    Book(
        title="Fifty Challenging Problems in Probability",
        authors=["Frederick Mosteller"],
        tags=["Maths", "Problem Solving"],
        average_rating=2,
        id=4
    ),
    Book(
        title="Algorithmic Puzzles",
        authors=["Anany Levitin", "Maria Levitin"],
        tags=["Problem Solving", "Computer Science", "Logic", "Interviews"],
        average_rating=4,
        id=5
    ),
    Book(
        title="How To Solve It",
        authors=["George Polya"],
        tags=["Maths", "Problem Solving", "Textbook"],
        average_rating=3,
        id=6
    ),
    Book(
        title="How To Solve It by Computer",
        authors=["Dromey"],
        tags=["Computer Science", "Logic", "Textbook"],
        average_rating=3,
        id=7
    ),
    Book(
        title="Fermat's Last Theorem",
        authors=["Simon Singh"],
        tags=["Maths", "History of Maths", "Biography"],
        average_rating=4.5,
        id=8
    ),
    Book(
        title="The Code Book",
        authors=["Simon Singh"],
        tags=["Maths", "History of Maths", "Computer Science"],
        average_rating=4,
        id=9
    ),
    Book(
        title="The Great Mathematical Problems",
        authors=["Ian Stewart"],
        tags=["Maths", "History of Maths", "Problem Solving"],
        average_rating=4,
        id=10
    ),
    Book(
        title="My Best Mathematical and Logic Puzzles",
        authors=["Martin Gardner"],
        tags=["Maths", "Logic", "Problem Solving"],
        average_rating=3,
        id=11
    ),
    Book(
        title="How to Think Like a Mathematician",
        authors=["Kevin Houston"],
        tags=["Maths", "Textbook", "Interviews"],
        average_rating=4.5,
        id=12
    ),
]

lib_size = len(book_lib)


def get_common_elements(list1: list, list2: list) -> list:
    """gets the the common elements between two lists

    Args:
        list1 (list): first list
        list2 (list): second list

    Returns:
        list: list of common elements
    """

    # find the intersection of the two sets
    common_elements = set(list1) & set(list2)

    return list(common_elements)


def count_common_elements(list1: list, list2: list) -> int:
    """counts the number of common elements between two lists

    Args:
        list1 (list): first list
        list2 (list): second list

    Returns:
        int: number of common elements
    """
    return len(get_common_elements(list1, list2))


def get_common_authors(book1: Book, book2: Book) -> list:
    """gets the common authors between two books

    Args:
        book1 (Book): first book
        book2 (Book): second book

    Returns:
        list: common authors
    """
    return get_common_elements(book1.authors, book2.authors)


def get_common_tags(book1: Book, book2: Book) -> list:
    """gets the common tags between two books

    Args:
        book1 (Book): first book
        book2 (Book): second book

    Returns:
        list: common tags
    """
    return get_common_elements(book1.tags, book2.tags)


def count_common_authors(book1: Book, book2: Book) -> int:
    """counts the number of common authors between two books

    Args:
        book1 (Book): first book
        book2 (Book): second book

    Returns:
        int: number of common authors
    """
    return len(get_common_authors(book1, book2))


def count_common_tags(book1: Book, book2: Book) -> int:
    """counts the number of common tags between two books

    Args:
        book1 (Book): first book
        book2 (Book): second book

    Returns:
        int: number of common tags
    """
    return len(get_common_tags(book1, book2))


def calculate_link_strength(book1: Book, book2: Book) -> int:
    """calculates the link strength between two books based on common authors
    and tags, each common tag adds 1 to the link strength, each common author
    adds 2 to the link strength

    Args:
        book1 (Book): first book
        book2 (Book): second book

    Returns:
        int: link strength between the two books
    """

    common_authors = count_common_authors(book1, book2)
    common_tags = count_common_tags(book1, book2)

    return (2 * common_authors) + common_tags


def create_link_matrix(library: list[Book]) -> np.ndarray:
    """creates a matrix which shows the link strength between all books

    Args:
        library (list[Book]): book library

    Returns:
        np.ndarray: link strength of all books
    """

    # create a null matrix
    matrix = np.zeros((lib_size, lib_size))

    # iterate through each book in the library where the column is greater
    # than the row which means it is a triangular matrix but then reflect it
    # across the diagonal to make it symmetric
    for i in range(lib_size):
        for j in range(lib_size):
            if j > i:
                # gets the link strength
                link_strength = calculate_link_strength(library[i], library[j])

                # apply it to 1 half of the matrix and its reflected half
                matrix[i][j] = link_strength
                matrix[j][i] = link_strength

    return matrix


def create_preferences_vector(library: list[Book],
                              preferences: list[str]) -> np.ndarray:
    """creates a vector which shows the number of common preferences for each
    book

    Args:
        library (list[Book]): book library
        preferences (list[str]): preferences

    Returns:
        np.ndarray: vector of common preferences for each book
    """
    # creates a null vector
    preferences_vector = np.zeros((lib_size, 1))

    # iterate through every book in the library
    for i, book in enumerate(library):
        num_common_preferences = count_common_elements(book.tags, preferences)
        num_common_authors = count_common_elements(book.authors, preferences)

        # each common tag adds 1 to the preferences vector, each common author
        # adds 2 to the preferences vector
        preferences_vector[i] = num_common_preferences + 2 * num_common_authors

    return preferences_vector


def create_rating_vector(library: list[Book]) -> np.ndarray:
    """creates a vector which shows the average rating of each book

    Args:
        library (list[Book]): book library

    Returns:
        np.ndarray: vector of average ratings for each book
    """
    # creates a null vector
    ratings_vector = np.zeros((lib_size, 1))

    # iterate through every book in the library
    for i, book in enumerate(library):
        ratings_vector[i] = book.average_rating

    return ratings_vector


def normalise_matrix(matrix: np.ndarray) -> np.ndarray:
    """normalises a 2d matrix by dividing each element by the column sum, if
    the column sum is 0, each element is made such that the sum is 1

    Args:
        matrix (np.ndarray): matrix to normalise

    Returns:
        np.ndarray: normalised matrix
    """
    row_sums = matrix.sum(axis=0, keepdims=True)

    row_sums[row_sums == 0] = 1

    return matrix / row_sums


# create a null vector to store the ranks of each book
R = np.zeros((lib_size, 1))

# create 3 matrices for the link strength, preferences and ratings
S = normalise_matrix(create_link_matrix(book_lib))
T = normalise_matrix(create_preferences_vector(book_lib, PREFERENCES))
U = normalise_matrix(create_rating_vector(book_lib))

# repeatedly calculate the ranks using a modified version of the PageRank
# formula
for _ in range(10000):
    R = (LINK_WEIGHT * S @ R) + (PREFERENCES_WEIGHT * T) + (RATINGS_WEIGHT * U)

ranks = {}

# create a dictionary of book titles to their respective ranks
for i, element in enumerate(R):
    ranks[book_lib[i].title] = element[0]

# sort the ranks descending
ranks = dict(
    sorted(ranks.items(),
           key=lambda item: item[1],
           reverse=True
           )
    )

# output the ranks
for i in ranks:
    print(f"{i}: {ranks[i]}")
