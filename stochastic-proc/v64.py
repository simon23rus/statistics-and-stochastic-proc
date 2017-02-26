def create_page_rank_markov_chain(links, damping_factor=0.15):
    ''' По веб-графу со списком ребер links строит матрицу
        переходных вероятностей соответствующей марковской цепи.
        
        links --- список (list) пар вершин (tuple),
        может быть передан в виде numpy.array, shape=(|E|, 2);
        damping_factor --- вероятность перехода не по ссылке (float);
        
        Возвращает prob_matrix --- numpy.matrix, shape=(|V|, |V|).
        '''
    
    links = np.array(links) # сделать список смежности r[i] = len(list_link[i])
    N = links.max() + 1  # Число веб-страниц
    link_list = [[] for i in range(N)]
    for elem in links:
        link_list[elem[0]].append(elem[1])
    
    prob_matrix = []
    for from_, page_links in tqdm(enumerate(link_list)):
        row = [1. / N] * N
        if len(page_links) == 0:
            prob_matrix.append(row)
            continue
        for to_ in range(len(link_list)):
            row[to_] = (1. - damping_factor) * ((1. / len(link_list[from_])) if to_ in page_links else 0) + damping_factor / N
        prob_matrix.append(row)
    return np.matrix(prob_matrix)


def page_rank(links, start_distribution, damping_factor=0.15,
              tolerance=10 ** (-7), return_trace=False):
    ''' Вычисляет веса PageRank для веб-графа со списком ребер links
        степенным методом, начиная с начального распределения start_distribution,
        доводя до сходимости с точностью tolerance.
        
        links --- список (list) пар вершин (tuple),
        может быть передан в виде numpy.array, shape=(|E|, 2);
        start_distribution --- вектор размерности |V| в формате numpy.array;
        damping_factor --- вероятность перехода не по ссылке (float);
        tolerance --- точность вычисления предельного распределения;
        return_trace --- если указана, то возвращает список распределений во
        все моменты времени до сходимости
        
        Возвращает:
        1). если return_trace == False, то возвращает distribution ---
        приближение предельного распределения цепи,
        которое соответствует весам PageRank.
        Имеет тип numpy.array размерности |V|.
        2). если return_trace == True, то возвращает также trace ---
        список распределений во все моменты времени до сходимости.
        Имеет тип numpy.array размерности
        (количество итераций) x |V|.
        '''
    
    prob_matrix = create_page_rank_markov_chain(links,
                                                damping_factor=damping_factor)
    distribution = np.matrix(start_distribution)
                                                
    last, current = distribution, np.dot(distribution, prob_matrix)
    trace = [last]
    while norm(current  - last) > tolerance:
        last, current = current, np.dot(current, prob_matrix)
        trace.append(last)



    if return_trace:
        return np.array(np.matrix(current)).ravel(), np.array(trace)
    else:
        return np.array(np.matrix(current)).ravel()
