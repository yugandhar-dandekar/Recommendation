import numpy as np

LINK_WEIGHT = 0.85
PREFERENCES_WEIGHT = 0.1
RATINGS_WEIGHT = 1 - (LINK_WEIGHT + PREFERENCES_WEIGHT)

PREFERENCES = ["Engineering"]


# library of books to be used for the recommendation system
library = [
    {
        "title": "Advanced Problems in Mathematics",
        "authors": ["Stephen Siklos"],
        "tags": ["Maths", "Problem Solving", "Oxbridge"],
        "avg_rating": 5,
    },
    {
        "title": "Stanford Mathematics Problem Book",
        "authors": ["George Polya", "Jeremy Kilpatrick"],
        "tags": ["Maths", "Problem Solving", "Oxbridge", "Interviews"],
        "avg_rating": 4,
    },
    {
        "title": "Professor Povey's Perplexing Problems",
        "authors": ["Thomas Povey"],
        "tags": ["Problem Solving", "Physics", "Engineering", "Interviews"],
        "avg_rating": 4,
    },
    {
        "title": "Fifty Challenging Problems in Probability",
        "authors": ["Frederick Mosteller"],
        "tags": ["Maths", "Problem Solving"],
        "avg_rating": 2,
    },
    {
        "title": "Algorithmic Puzzles",
        "authors": ["Anany Levitin", "Maria Levitin"],
        "tags": ["Problem Solving", "Computer Science", "Logic", "Interviews"],
        "avg_rating": 4,
    },
    {
        "title": "How To Solve It",
        "authors": ["George Polya"],
        "tags": ["Maths", "Problem Solving", "Textbook"],
        "avg_rating": 3,
    },
    {
        "title": "How To Solve It by Computer",
        "authors": ["Dromey"],
        "tags": ["Computer Science", "Logic", "Textbook"],
        "avg_rating": 3,
    },
    {
        "title": "Fermat's Last Theorem",
        "authors": ["Simon Singh"],
        "tags": ["Maths", "History of Maths", "Biography"],
        "avg_rating": 4.5,
    },
    {
        "title": "The Code Book",
        "authors": ["Simon Singh"],
        "tags": ["Maths", "History of Maths", "Computer Science"],
        "avg_rating": 4,
    },
    {
        "title": "The Great Mathematical Problems",
        "authors": ["Ian Stewart"],
        "tags": ["Maths", "History of Maths", "Problem Solving"],
        "avg_rating": 4,
    },
    {
        "title": "My Best Mathematical and Logic Puzzles",
        "authors": ["Martin Gardner"],
        "tags": ["Maths", "Logic", "Problem Solving"],
        "avg_rating": 3,
    },
    {
        "title": "How to Think Like a Mathematician",
        "authors": ["Kevin Houston"],
        "tags": ["Maths", "Textbook", "Interviews"],
        "avg_rating": 4.5,
    },
]


# normalising a matrix by dividing each column by its column sun
# column sums that are 0 are replace with 1/rows so that eeach column sums to 1
def normalise_matrix(matrix):
    # get the dimensions of the matrix
    columns = matrix.shape[1]
    rows = matrix.shape[0]

    # for each column in the matrix
    for col in range(columns):
        # extract the column
        matrix_column = matrix[:, col]

        # calculate the sum of the given column
        column_sum = matrix_column.sum()

        # check if the sum is or is not 0
        if column_sum != 0:
            # in this case where there is at least one non-zero element,
            # every element is divided by the sum of the column
            matrix[:, col] = matrix_column / column_sum
        else:
            # in this case where all ements are 0, ever element is replaced
            # with 1/rows so that the column sums to 1
            matrix[:, col] = 1 / rows

    return matrix


# gets the common authors between two books
def get_common_authors(book1_index, book2_index, booklist):
    # extract the authors of the two books
    authors1 = booklist[book1_index]["authors"]
    authors2 = booklist[book2_index]["authors"]

    # find the intersection of the authors
    common_authors = list(set(authors1).intersection(set(authors2)))

    return common_authors


# gets common tags between 2 books
def get_common_tags(book1_index, book2_index, booklist):
    # extract the common tags
    authors1 = booklist[book1_index]["tags"]
    authors2 = booklist[book2_index]["tags"]

    # find the intersection of the tags
    common_authors = list(set(authors1).intersection(set(authors2)))

    return common_authors


# gets the strength of the link between two books
def get_link_strength(book1_index, book2_index, booklist):
    # extract the authors and tags of the two books
    common_authors = get_common_authors(book1_index, book2_index, booklist)
    common_tags = get_common_tags(book1_index, book2_index, booklist)

    # calculate the link strength, each common tag is +1 to the strength,
    # each common author is +2 to the strength
    link_strength = len(common_authors) * 2 + len(common_tags)

    return link_strength


# gets a matrix of links between all books
def get_link_matrix(booklist, normalise=True):
    # create a null matrix
    matrix = np.zeros((len(booklist), len(booklist)), dtype=float)

    # iterate through each row and column of the matrix
    for row in range(len(booklist)):
        for col in range(len(booklist)):
            # ensures that the column is greated that the row to reduce the
            # number of strngth calculations as the matrix can be reflected
            # diagonally to become symmetric, it does not matter wether col >
            # row or row > col
            if col > row:
                strength = get_link_strength(row, col, booklist)

                # reflect the matrix diagonally
                matrix[row][col] = strength
                matrix[col][row] = strength

    return normalise_matrix(matrix) if normalise else matrix


# gets a vector of tags for each book
def get_tag_vector(booklist, tags, normalise=True):
    # create a null matrix
    matrix = np.zeros((len(booklist), 1), dtype=float)

    # iterate through each row of the matrix
    for i, book in enumerate(booklist):
        authors_and_tags = book["tags"] + book["authors"]

        # find the intersection of the authors and tags between the 2 books
        common_items = set(authors_and_tags).intersection(set(tags))

        matrix[i][0] = len(common_items)

    return normalise_matrix(matrix) if normalise else matrix


# gets a vector of ratings for each book
def get_rating_vector(booklist, normalise=True):
    # create a null matrix
    matrix = np.zeros((len(booklist), 1), dtype=float)

    # iterate through each row of the matrix
    for i, book in enumerate(booklist):
        matrix[i][0] = book["avg_rating"]

    return normalise_matrix(matrix) if normalise else matrix


def main():
    # generate the 3 matrices that will be used to calculate the rank
    link_matrix = get_link_matrix(library)
    tag_matrix = get_tag_vector(library, PREFERENCES)
    rating_matrix = get_rating_vector(library)

    # unordered ranks of books, the index corresponds to the book in the
    # library
    rank_vector = np.ones((len(library), 1)) / len(library)

    # iteratively updates the rating vector based on the 3 matrices
    for _ in range(10000):
        rank_vector = (
            LINK_WEIGHT * link_matrix @ rank_vector
            + PREFERENCES_WEIGHT * tag_matrix
            + RATINGS_WEIGHT * rating_matrix
        )

    ranks = {}

    # creates a dictionary of book titles and their corresponding ranks
    for i, element in enumerate(rank_vector):
        ranks[library[i]["title"]] = element[0]

    # sorts the dictionary by value in descending order
    ranks = dict(sorted(ranks.items(), key=lambda item: item[1], reverse=True))

    # prints the ranks of the books
    for i in ranks:
        print(f"{i}: {ranks[i]}")


if __name__ == "__main__":
    main()
