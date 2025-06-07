"""
__future__ allows using class names before they are fully defined
numpy allows efficient matrix operations
"""
from __future__ import annotations

import numpy as np


class Book():
    """
    Object representation of a book
    """

    def __init__(self, title: str, authors: list[str], tags: list[str],
                 average_rating: float, book_id: int) -> None:

        # set book attributes
        self.title = title
        self.authors = authors
        self.tags = tags
        self.average_rating = average_rating
        self.id = book_id

    def count_common_authors(self, other: Book
                             ) -> int:
        """counts the common authors between two books

        Args:
            other (Book): other book to compare \
                with

        Returns:
            int: number of common authors
        """
        return len(set(self.authors) & set(other.authors))

    def count_common_tags(self, other: Book
                          ) -> int:
        """counts the number of common tags between two books

        Args:
            other (Book): other book to compare \
                with

        Returns:
            int: number of common tags
        """
        return len(set(self.tags) & set(other.tags))

    def calculate_link_strength(self, other: Book
                                ) -> int:
        """calculates the link strength between two books, authors are \
            weighted at 2, tags weighted at 1

        Args:
            other (Book): other book to compare \
                with

        Returns:
            int: link strength between the two books
        """
        return (2 * self.count_common_authors(other)) + \
            self.count_common_tags(other)


class RecommendationAlgorithm():
    """Book recommendation algorithm which recommends books to the user
    """

    def __init__(self, library: list[Book],
                 preferences: list[str],
                 default_pagerank_iterations: int = 10_000,
                 link_weight: float = 0.85,
                 preferences_weight: float = 0.1,
                 ) -> None:

        # set library data
        self.__library = library
        self.__LIB_SIZE = len(library)

        # set the user preferences
        self.__PREFERENCES = preferences

        # set the PageRank iteraitons
        self.PAGERANK_ITERATIONS = default_pagerank_iterations

        # set the weights for the PageRank algorithm
        self.LINK_WEIGHT = link_weight
        self.PREFERENCES_WEIGHT = preferences_weight
        self.RATINGS_WEIGHT = 1 - (self.LINK_WEIGHT + self.PREFERENCES_WEIGHT)

    def normalise_matrix(self, matrix: np.ndarray) -> np.ndarray:
        """normalises a matrix to be stochastic, meaning that the sum of each \
            column is 1

        Args:
            matrix (np.ndarray): matrix to normalise

        Returns:
            np.ndarray: normalised matrix
        """

        # uses the column axis to calculate the sums, keepdims ensures that
        # the resulting sums have the same dimensions as the original vector

        # converts the matrix to float to avoid errors caused by division
        # matrix is copied to avoid modifying the original matrix
        matrix = matrix.astype(float).copy()

        # calculate the initial sum of each column in the matrix
        col_sums = matrix.sum(axis=0, keepdims=True)

        # get boolean array of columns with sums of 0
        zero_col_sums = (col_sums == 0)[0]

        # if there is any column with a sum of 0, set the values in that
        # column to 1 so that they can be properly normalised
        if zero_col_sums.any():
            # set the values to 1
            matrix[:, zero_col_sums] = 1

        # recalculate the column sums after setting the zero columns to 1
        col_sums = matrix.sum(axis=0, keepdims=True)

        # normalises the matrix to be stochastic
        normalised_matrix = matrix / col_sums

        return normalised_matrix

    def create_link_matrix(self) -> np.ndarray:
        """creates an adjacency matrix which shows the link strength between \
            all books in the
        library

        Returns:
            np.ndarray: 2D array representing the link strength between all \
                books
        """

        # create a null matrix
        matrix = np.zeros(
            (self.__LIB_SIZE, self.__LIB_SIZE),
            dtype=int
            )

        # the size of the 2D matrix will be equal to the number of books in
        # the library, hence, we can iterate through each book in the library
        # if the index of the second book is greater than the index of the
        # first book, calculate the link strength and then add it to the
        # adjacency matrix at the corresponding indices.
        # Visually, this is exactly the same as creating an upper triangular
        # matrix, without the diagonal, and then reflecting it across the
        # diagonal to create a symmetric adjacency matrix
        for book1_index, book1 in enumerate(self.__library):
            for book2_index, book2 in enumerate(self.__library):
                # skip if the second book index is not greater than the first
                # which means that the column index is less than the row index
                # which means that it is not in the upper triangular matrix
                # (ignoring the diagonal)
                if not book2_index > book1_index:
                    continue

                # calculate the link strength between the two books
                link_strength = book1.calculate_link_strength(book2)

                # add the link strength to the matrix at the corresponding
                # index and reflect it
                matrix[book1_index][book2_index] = link_strength
                matrix[book2_index][book1_index] = link_strength

        return matrix

    def count_common_elements(self, list1: list, list2: list) -> int:
        """counts the number of common elements between two lists

        Args:
            list1 (list): list 1
            list2 (list): list 2

        Returns:
            int: number of common elements
        """

        return len(set(list1) & set(list2))

    def create_rating_vector(self) -> np.ndarray:
        """creates a vector which contains the average rating of each book

        Returns:
            np.ndarray: vector of average ratings of each book
        """

        # creates a null vector
        ratings_vector = np.zeros(self.__LIB_SIZE)

        # iterate through every book in the library and add the average rating
        # of the book to the corresponding index
        for book_index, book in enumerate(self.__library):
            ratings_vector[book_index] = book.average_rating

        return ratings_vector

    def create_preferences_vector(self) -> np.ndarray:
        """creates a vector which shows the number of common preferences for \
            each book

        Returns:
            np.ndarray: vector of commong preferences for each book
        """

        # create a null vector
        preferences_vector = np.zeros(self.__LIB_SIZE)

        # iterate through every book in the library and find the number of
        # common preferences. It is important to note that here the authors
        # are not weighted at 2 because we have already done this in the link
        # matrix and we give more important to that then the preferences
        for book_index, book in enumerate(self.__library):
            # count the common tags
            common_tags = self.count_common_elements(book.tags,
                                                     self.__PREFERENCES)

            # count the common authors
            common_authors = self.count_common_elements(book.authors,
                                                        self.__PREFERENCES)

            preferences_vector[book_index] = common_tags + common_authors

        return preferences_vector

    def calculate_rankings(self) -> np.ndarray:
        """calculates the rankings of each book based on PageRank

        Returns:
            np.ndarray: book rankings
        """
        rankings = np.zeros(self.__LIB_SIZE)

        # create stochastic matrices for links, preferences and ratings for
        # the algorithm
        S = self.normalise_matrix(self.create_link_matrix())
        T = self.normalise_matrix(self.create_preferences_vector())
        R = self.normalise_matrix(self.create_rating_vector())

        # iteratively calculate the rankings
        for _ in range(self.PAGERANK_ITERATIONS):
            rankings = (self.LINK_WEIGHT * S @ rankings) + \
                (self.PREFERENCES_WEIGHT * T) + (self.RATINGS_WEIGHT * R)

        return rankings

    def get_recommendations(self) -> list[Book]:
        """gets the recommended books based on the rankings

        Returns:
            list[Book]: list of recommended books
        """
        rankings = self.calculate_rankings()

        # get the indices of the books sorted by their rankings in descending
        # order
        sorted_indices = np.argsort(rankings)[::-1]

        # return the books in the order of their rankings
        return [self.__library[i].title for i in sorted_indices]


# our book library
books = [
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

# instantiate the recommendation algorithm with the library and preferences
alg = RecommendationAlgorithm(
    library=books,
    preferences=['Engineering', 'Physics', 'Problem Solving']
    )

# get top 5 recommended books
top_books = alg.get_recommendations()[:5]

[print(f"{position + 1}: {book}") for position, book in enumerate(top_books)]
