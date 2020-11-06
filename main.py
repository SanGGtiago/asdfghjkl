def cal_cosine_similarity(v1, v2):
    '''
    calculate the cosine similarity
    :param v1: python list
    :param v2: python list
    :return: int
    '''
    if len(v1) != len(v2) or len(v1) == 0 or len(v2) == 0:
        print("Error: invalid input, the length of vectors should be equal and larger than 0")
        return

    result = 0.0
    numerator = 0.0
    denominator = 0.0

    # numerator = np.dot(v1, v2)
    # denominator = np.sqrt(np.dot(v1, v1)) * np.sqrt(np.dot(v2, v2))

    numerator = np.inner(v1, v2)
    denominator = np.linalg.norm(v1) * np.linalg.norm(v2)

    if denominator != 0.0:
        result = numerator / denominator

    return result

def find_similar_neighbor_cosine(user_id, train_matrix, test_map):
    '''
    find the top k neighbors with cosine similarity
    :param user_id: int
    :param train_matrix: python 2-d array
    :param test_map: python dictionary
    :return: list_of_tuple(user_id, cosine_similarity)
    '''

    test_user = test_map[user_id]
    list_of_unrated_movie = test_user.get_list_of_unrated_movie()
    list_of_rated_movie = test_user.get_list_of_rated_movie()
    list_of_rate_of_rated_movie = test_user.get_list_of_rate_of_rated_movie()

    list_of_neighbor = []

    # find the neighbor
    # go through the 200 users
    for row in range(len(train_matrix)):
        train_user_id = row + 1

        # skip the target user itself
        if train_user_id == user_id: continue

        common_movie = 0

        numerator = 0.0
        denominator = 0.0
        cosine_similarity = 0.0

        test_vector = []
        train_vector = []

        for i in range(len(list_of_rated_movie)):
            movie_id = list_of_rated_movie[i]
            test_movie_rate = list_of_rate_of_rated_movie[i]
            train_movie_rate = train_matrix[train_user_id - 1][movie_id - 1]

            # movie rate with zero means the user doesn't rate the movie
            # we should not consider it as part of the calculation of cosine similarity
            if train_movie_rate != 0:
                common_movie += 1

                test_vector.append(test_movie_rate)
                train_vector.append(train_movie_rate)

        # the common movie between test data and train data should be larger than 1
        if common_movie > 1:
            cosine_similarity = cal_cosine_similarity(test_vector, train_vector)

            list_of_neighbor.append((train_user_id, cosine_similarity))
        if common_movie < 2:
            list_of_neighbor.append([0])

    #list_of_neighbor.sort(key=lambda tup : tup[1], reverse=True)

    return list_of_neighbor

def cal_pearson_correlation(test_users, test_mean_rate, train_users, train_mean_rate):
    '''
    calculate the pearson correlation
    :param test_users: list of tuple(movie id, movie rate) from unrated movie in test data
    :param test_mean_rate: mean of rate in the test user
    :param train_users: python list
    :param train_mean_rate: mean of rate in the train user
    :return: float
    '''
    # there are 3 cases:
    # case 1: result > 0
    #           return result
    # case 2.1: result == 0, because of no common component
    #           return 0
    # case 2.2: result == 0, because of in one or two of vector, each component == mean of vector
    #           return the mean of vector

    numerator = 0.0
    denominator = 0.0

    # filter the common components as vector
    vector_test_rates = []
    vector_train_rates = []

    for test_movie_id, test_movie_rate in test_users:
        train_movie_rate = train_users[test_movie_id - 1]
        if train_movie_rate > 0 and test_movie_rate > 0:
            vector_test_rates.append(test_movie_rate)
            vector_train_rates.append(train_movie_rate)

    # no common component
    if len(vector_train_rates) == 0 or len(vector_test_rates) == 0:
        return 0.0

    adj_vector_test_users = [movie_rate - test_mean_rate for movie_rate in vector_test_rates]
    adj_vector_train_users = [movie_rate - train_mean_rate for movie_rate in vector_train_rates]

    numerator = np.inner(adj_vector_train_users, adj_vector_train_users)
    denominator = np.linalg.norm(adj_vector_test_users) * np.linalg.norm(adj_vector_train_users)

    # each component of one or both vectors is the same
    if denominator == 0.0:
        return 0.0

    return numerator / denominator

def find_similar_neighbor_pearson(user_id, train_matrix, test_map, train_mean_rate_map):
    '''
    calculate all neighbors's pearson correlation for given test user
    :param user_id: int
    :param train_df: python 2-d array
    :param test_map: python dictionary
    :param train_mean_rate_map: python dictionary, the mean of each train user
    :return: a list of tuple(user id, similarity)
    '''
    list_of_neighbors = []

    # average rate of given test user
    avg_movie_rate_in_test = avg_movie_rate_of_test_user(user_id, test_map)

    user = test_map[user_id]
    list_of_rated_movie = user.get_list_of_rated_movie()
    list_of_rate_of_rated_movie = user.get_list_of_rate_of_rated_movie()
    list_of_unrated_movie = user.get_list_of_unrated_movie()

    # zipped_list_of_rated_movie_with_rate = list(zip(list_of_rated_movie, list_of_rate_of_rated_movie))
    zipped_list_of_rated_movie_with_rate = []
    for i in range(len(list_of_rated_movie)):
        zipped_list_of_rated_movie_with_rate.append((list_of_rated_movie[i], list_of_rate_of_rated_movie[i]))

    # find the neighbor
    # go through the 200 users
    for index, row in enumerate(train_matrix):
        train_user_id = index + 1

        # skip the target user
        if train_user_id == user_id: continue

        # average rate of given train user
        avg_movie_rate_in_train = train_mean_rate_map[train_user_id]

        pearson_correlation = cal_pearson_correlation(zipped_list_of_rated_movie_with_rate, avg_movie_rate_in_test, row, avg_movie_rate_in_train)

        # correct the pearson correlation
        # range is [-1, 1]
        if pearson_correlation > 1.0:
            pearson_correlation = 1.0
        if pearson_correlation < -1.0:
            pearson_correlation = -1.0

        if pearson_correlation != 0.0:
            list_of_neighbors.append((train_user_id, pearson_correlation))

        if pearson_correlation == 0.0:
            list_of_neighbors.append([0.0])

    # list_of_neighbors.sort(key=lambda tup: tup[1], reverse=True)

    return list_of_neighbors

def cal_adjusted_cosine_similarity(target_id, neighbor_id, t_train_matrix, train_mean_rate_map):
    '''
    calculate the adjusted cosine similarity in item-based approach
    :param target_id: int, the given movie id
    :param neighbor_id: the certain movie id in the train data
    :param t_train_matrix: the transposed train matrix
    :param train_mean_rate_map: mean rate of each user in the train data
    :return: float
    '''
    adj_cosine_sim = 0.0

    target_row = t_train_matrix[target_id - 1]
    neighbor_row = t_train_matrix[neighbor_id - 1]

    # filter the common component
    # subtract the mean rate
    target_vector = []
    neighbor_vector = []
    for i in range(len(t_train_matrix[0])):
        if target_row[i] > 0 and neighbor_row[i] > 0:
            target_vector.append(target_row[i] - train_mean_rate_map[i + 1])
            neighbor_vector.append((neighbor_row[i]) - train_mean_rate_map[i + 1])

    # calculate the adjusted cosine similarity
    # the common component should be larger than 1
    if len(target_vector) == len(neighbor_vector) and len(target_vector) > 1:
        adj_cosine_sim = cal_cosine_similarity(target_vector, neighbor_vector)

    return adj_cosine_sim

def build_map_similar_neighbor_adj_cosine(train_matrix, train_mean_rate_map):
    '''
    build map of the similar neighbors based on adjusted cosine similarity
    :param movie_id: int
    :param train_matrix: python 2-d array
    :param train_mean_rate_map: python dictionary, K: train user id, V: mean rate of given train user
    :return: python dictionary, K: movie id, V: python dictionary(K: movie id, V: similarity)
    '''
    neighbor_map = {}

    # transpose the train matrix
    t_train_matrix = np.array(train_matrix).T
    print(len(t_train_matrix))
    for i in range(len(t_train_matrix)):
        target_id = i + 1

        if target_id not in neighbor_map:
            neighbor_map[target_id] = {}

        for j in range(i + 1, len(t_train_matrix)):
            neighbor_id = j + 1

            adj_cosine_sim = cal_adjusted_cosine_similarity(target_id, neighbor_id, t_train_matrix, train_mean_rate_map)

            neighbor_map[target_id][neighbor_id] = adj_cosine_sim

            if neighbor_id not in neighbor_map:
                neighbor_map[neighbor_id] = {}

            neighbor_map[neighbor_id][target_id] = adj_cosine_sim

    return neighbor_map

