import heapq
import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as splinalg
from scipy.io import mmread
import gzip
import tarfile
import os
import time


class Node:
    def __init__(self, index):
        self.index = index
        self.result = 0.0
        self.values = []
        self.valueindexes = []
        self.dependencies = set()
        self.dependents = []
        self.cost = 0.0
        self.dependant_from_lower = 0
        self.move = False
        self.moved = False

    def __repr__(self):
        return f"Node(index={self.index}, result={self.result}, dependencies={self.dependencies}, dependents={self.dependents}, values={self.values}, valueindexes={self.valueindexes}, cost={self.cost})"


def read_sparse_matrix_from_gz(file_name):
    file_path = os.path.join(os.getcwd(), file_name)
    with gzip.open(file_path, 'rt') as f:
        matrix = mmread(f)
        csr = matrix.tocsr()
        return csr


def extract_mtx_files_from_tar_gz(archive_path):
    mtx_files = []
    with tarfile.open(archive_path, 'r:gz') as tar:
        for member in tar.getmembers():
            if member.isfile() and member.name.endswith('.mtx'):
                tar.extract(member)
                mtx_files.append(member.name)
    return mtx_files


def node_analysis(nodes, l):
    global levels_with_object
    levels_with_object = []
    levels = []
    processed_nodes = set()
    root_nodes = []
    indexes = []

    ##obje için
    root_nodes_with_object = []
    processed_nodes_with_object = []








    for node in nodes:
        if not node.dependencies:
            root_nodes.append(node.index)
            processed_nodes.add(node.index)
            indexes.append(node.index)

            #obje
            root_nodes_with_object.append(node)

    levels.append(root_nodes)
    levels_with_object.append(root_nodes_with_object)

    while len(processed_nodes) < l:
        temp_current_level_objects = []
        temp_processed_nodes = []
        temp_indexes = []
        current_level = []

        for node in nodes:
            if node.index not in processed_nodes:
                is_not_dependent = True

                for dep in node.dependencies:
                    if dep not in indexes:
                        is_not_dependent = False
                        break

                if is_not_dependent:
                    current_level.append(node.index)
                    temp_current_level_objects.append(node)
                    temp_indexes.append(node.index)
                    temp_processed_nodes.append(node.index)

        levels.append(current_level)
        levels_with_object.append(temp_current_level_objects)

        for i in temp_processed_nodes:
            processed_nodes.add(i)
        for i in temp_indexes:
            indexes.append(i)


    return levels

def cache_execution(output_widget):
    global levels_with_object
    cache = cache_subexpressions(levels_with_object)

    length = len(cache)
    output_widget.insert(tk.END,
                         f"\nThe expressions are analyzed and {length} number of elements found and stored in cache as most frequent \n")


# Cache methodu
def cache_subexpressions(levels_with_object, cache_size_factor=0.1):
    """
    levels_with_object içindeki Node'ların en sık kullanılan subexpression'larını tespit edip cache'ler.
    :param levels_with_object: Node'ların bulunduğu seviyeler listesi
    :param cache_size_factor: Cache büyüklüğünü belirlemek için matris boyutuyla çarpan
    :return: En sık kullanılan işlemlerle dolu bir cache
    """

    # Alt ifadelerin frekansını tutmak için bir sözlük
    subexpression_frequency = {}

    # Tüm Node'lardaki işlemleri incele
    for level in levels_with_object:
        for node in level:
            # "Values" ve "ValueIndexes" kullanılarak matematiksel işlemi oluştur
            values = node.values  # Çarpılacak değerler
            value_indexes = node.valueindexes  # Hangi vektörlerle çarpılacaklar

            # Alt ifadeleri oluştur (string ya da tuple olarak)
            subexpression = tuple((values[i], value_indexes[i]) for i in range(len(values)))

            # Frekansı güncelle
            if subexpression in subexpression_frequency:
                subexpression_frequency[subexpression] += 1
            else:
                subexpression_frequency[subexpression] = 1

    # Matris büyüklüğüne göre cache boyutunu belirle
    total_expressions = len(subexpression_frequency)
    cache_size = max(1, int(total_expressions * cache_size_factor))  # En az 1 eleman cache'lenir

    # Frekansa göre sıralayıp en sık kullanılan subexpression'ları al
    sorted_expressions = sorted(subexpression_frequency.items(), key=lambda item: item[1], reverse=True)
    cache = {expr[0]: expr[1] for expr in sorted_expressions[:cache_size]}

    return cache


def trouble_shoot(output_widget):
    global levels_costs
    global average_cost
    global levels_with_object

    calculate_level_costs()

    output_widget.insert(tk.END,
                         f"The main avg cost: {average_cost} \n")

    cost = 0
    temp_count = 0
    avg_cost = 0
    for level in levels_with_object:
        temp_count += 1
    for level in levels_with_object:
        for node in level:
            cost += node.cost
    avg_cost = cost / temp_count



    output_widget.insert(tk.END, f"The manuel main avg cost: {avg_cost} \n")






def deneme(output_widget):
    global levels_costs  # Cost of the levels in the array format
    global average_cost  # Calculated avg cost
    global levels_with_object  # Array with the node objects in levels

    start_time = time.time()

    total_rewritten_nodes = 0

    start_level_size = len(levels_with_object)

    temp_levels_with_object = levels_with_object  # Using temp_levels_with_object to access node objects in levels

    output_widget.delete(1.0, tk.END)  # Clear previous output before displaying levels

    repeated_merges = 0

    # Iterate through levels in reverse order, starting from the last level
    for level_idx in range(len(levels_with_object) - 1, 1, -1):
        current_level = levels_with_object[level_idx]  # Current level in iteration
        next_level = levels_with_object[level_idx - 1]  # Next level in iteration



        # Average cost'un %45'inden düşük costtaki levelları mergeleme
        if levels_costs[level_idx] < average_cost * 0.45:  # %45
            if levels_costs[level_idx] + levels_costs[level_idx - 1] < average_cost * 1.3:
                if repeated_merges > 5:
                    repeated_merges = 0
                    continue
                nodes_to_move = []  # To hold the nodes to be moved up
                repeated_merges +=1

                # Loop through node objects in the current level

                for node in current_level:
                    temp_cost = 0


                    # Update dependencies for the node being moved
                    dependencies_to_be_removed = set()
                    new_dependencies = set()  # Collect new dependencies to be added
                    for dependency in node.dependencies:  # Checking every dependency for the node to be moved
                        keep_dependency = True
                        for dependant in levels_with_object[level_idx - 1]:  # Checking dependency of every node in upper level
                            if dependency == dependant.index:
                                keep_dependency = False
                                dependencies_to_be_removed.add(dependency)  # Mark dependency for removal


                                for dependant_value in dependant.values:
                                    node.values.append(dependant_value)

                                for dependant_dependency in dependant.dependencies:
                                    new_dependencies.add(dependant_dependency)  # Collect new dependencies to add

                    # After finishing iteration over dependencies, remove old dependencies
                    for dependency_to_remove in dependencies_to_be_removed:
                        node.dependencies.remove(dependency_to_remove)

                    # Now add the new dependencies that we collected
                    node.dependencies.update(new_dependencies)
                    nodes_to_move.append(node)  # Appending the node to be moved

                # Move nodes to the upper level (merge them into the next level)
                for node in nodes_to_move:
                    if node.moved == False:
                        total_rewritten_nodes+=1
                        node.moved = True

                    next_level.append(node)
                    levels_costs[level_idx - 1] += node.cost

                # Remove nodes from the current level after moving
                for node in nodes_to_move:
                    current_level.remove(node)
                    levels_costs[level_idx] -= node.cost

                # Ensure that merged levels are correctly maintained
                if not current_level:  # If the current level becomes empty, shift the levels up
                    # We only need to shift levels if there are empty levels
                    if level_idx < len(levels_with_object) - 1:  # Check if there are levels to move
                        # Shift the next levels up
                        for i in range(level_idx, len(levels_with_object) - 1):
                            levels_with_object[i] = levels_with_object[i + 1]  # Move the next level up

                        # The last level is now empty, so we delete it
                        levels_with_object.pop()  # Remove the last level after shifting
                # Recalculate level costs after merging
                calculate_level_costs()

            else:
                repeated_merges = 0
                continue
        else:
            repeated_merges = 0
            continue


                #burdan itibaren stage2, priority queue

    # Priority Queue oluşturma
    priority_levels = []  # Final priority-based levels array

    for level_idx, level in enumerate(levels_with_object):
        priority_queue = []  # Temporary heap for the current level
        for node in level:
            # Kendisine bağlı olan node'ların sayısını hesapla
            dependency_count = sum(1 for upper_level in levels_with_object[level_idx + 1:]
                                   for n in upper_level if node.index in n.dependencies)
            # Priority queue için tuple olarak ekle (öncelik değeri, node.index, node)
            heapq.heappush(priority_queue, (-dependency_count, node.index, node))

        # Priority queue'yu sıralı bir listeye çevir
        sorted_nodes = [heapq.heappop(priority_queue)[2] for _ in range(len(priority_queue))]
        priority_levels.append(sorted_nodes)  # Bu levelın sıralı nodelarını ekle




    calculate_level_costs()

    output_widget.insert(tk.END, f"Average cost:{average_cost}\n")

    most_dependent_nodes = get_most_dependent_nodes(priority_levels)  #get most dependent node indexes in levels

    # burdan sonra stage2: priority queue kullanarak taşıma işlemi








    update_dependant_from_lower()
    queue=update_queue(levels_with_object)

    for level_idx in range(len(levels_with_object)-1 , 1, -1):
        update_dependant_from_lower()
        queue = update_queue(levels_with_object)

        current_level = levels_with_object[level_idx]
        next_level = levels_with_object[level_idx - 1]

        nodes_to_move = []

        for node in current_level:
            # Update dependencies for the node being moved
            dependencies_to_be_removed = set()
            new_dependencies = set()  # Collect new dependencies to be added
            if queue[level_idx - 1] in node.dependencies:
                for dependant in next_level:
                    if dependant.index == queue[level_idx - 1]:

                        dependencies_to_be_removed.add(dependant.index)

                        for dependency in dependant.dependencies:
                            new_dependencies.add(dependency)

                        # After finishing iteration over dependencies, remove old dependencies
                        for dependency_to_remove in dependencies_to_be_removed:
                            node.dependencies.remove(dependency_to_remove)

                        # Now add the new dependencies that we collected
                        node.dependencies.update(new_dependencies)

                        nodes_to_move.append(node)  # Appending the node to be moved
                        node.move = True

        # Move nodes to the upper level
        for node in nodes_to_move:
            if node.moved == False:
                total_rewritten_nodes+=1
                node.moved = True

            next_level.append(node)
            levels_costs[level_idx - 1] += node.cost

        temp_level = []

        for node in current_level:
            if node.move:
                continue
            else:
                temp_level.append(node)

        levels_with_object[level_idx] = temp_level

        for node in next_level:
            node.move = False

        for node in current_level:
            node.move = False
    current_level = levels_with_object[len(levels_with_object)-1]
    for level_idx in range(len(levels_with_object) - 1, 1, -1):
        current_level = levels_with_object[level_idx]  # Current level in iteration
        if not current_level:
            levels_with_object.pop()
        else:
            break

    for level_idx in range(len(levels_with_object) - 1, 1, -1):

        current_level = levels_with_object[level_idx]  # Current level in iteration
        # Ensure that merged levels are correctly maintained
        # Ensure that merged levels are correctly maintained
        if not current_level:  # If the current level becomes empty, shift the levels up
            # We only need to shift levels if there are empty levels
            if level_idx < len(levels_with_object) - 1:  # Check if there are levels to move
                # Shift the next levels up
                for i in range(level_idx, len(levels_with_object) - 1):
                    levels_with_object[i] = levels_with_object[i + 1]  # Move the next level up

                # The last level is now empty, so we delete it
                levels_with_object.pop()  # Remove the last level after shifting


    end_time = time.time()
    elapsed_time = end_time - start_time

    final_level_size = len(levels_with_object)



    #  print the final result
    temp_increment_for_level = 0
    for level in levels_with_object:
        temp_level = []
        # Her seviyedeki node'ları al ve index'lerini ekle
        for node in level:
            temp_level.append(node.index)

        # Burada her seviyeyi düzgün yazdırıyoruz
        output_widget.insert(tk.END, f"Level {temp_increment_for_level}: {temp_level}\n")
        temp_increment_for_level += 1

    output_widget.insert(tk.END, f"Starting level size :{start_level_size}, Final level size:{final_level_size}\n")
    output_widget.insert(tk.END, f"Level size reduced by :{start_level_size - final_level_size} levels, which is by %{100-(final_level_size/start_level_size)*100}\n")
    output_widget.insert(tk.END, f"total rewritten nodes  :{total_rewritten_nodes}\n")

    output_widget.insert(tk.END, f"Elapsed Time: {elapsed_time}\n")




def unoptimized_deneme(output_widget):
    global levels_costs  # Cost of the levels in the array format
    global average_cost  # Calculated avg cost
    global levels_with_object  # Array with the node objects in levels

    start_time = time.time()
    start_level_size = len(levels_with_object)



    temp_levels_with_object = levels_with_object  # Using temp_levels_with_object to access node objects in levels

    output_widget.delete(1.0, tk.END)  # Clear previous output before displaying levels

    # Iterate through levels in reverse order, starting from the last level
    for level_idx in range(len(levels_with_object) - 1, 1, -1):
        cost = 0
        temp_count = 0
        avg_cost = 0
        for level in levels_with_object:
            temp_count += 1
        for level in levels_with_object:
            for node in level:
                cost+= node.cost
        avg_cost = cost/len(levels_with_object)

        current_level = levels_with_object[level_idx]  # Current level in iteration
        next_level = levels_with_object[level_idx - 1]  # Next level in iteration


        current_cost = 0
        next_cost = 0
        for node in levels_with_object[level_idx]:
            current_cost += node.cost

        for node in levels_with_object[level_idx -1]:
            next_cost += node.cost

        # Average cost'un %45'inden düşük costtaki levelları mergeleme
        if current_cost < avg_cost * 0.45:  # %45
            if current_cost + next_cost < avg_cost * 1.3:
                nodes_to_move = []  # To hold the nodes to be moved up

                # Loop through node objects in the current level
                for node in current_level:
                    output_widget.insert(tk.END,
                                         f"The node: {node.index} is rewritten from level: {level_idx} to level {level_idx - 1}\n")

                    # Update dependencies for the node being moved
                    dependencies_to_be_removed = set()
                    new_dependencies = set()  # Collect new dependencies to be added
                    for dependency in node.dependencies:  # Checking every dependency for the node to be moved
                        keep_dependency = True
                        for dependant in levels_with_object[level_idx - 1]:  # Checking dependency of every node in upper level
                            if dependency == dependant.index:
                                keep_dependency = False
                                dependencies_to_be_removed.add(dependency)  # Mark dependency for removal

                                for dependant_dependency in dependant.dependencies:
                                    new_dependencies.add(dependant_dependency)  # Collect new dependencies to add

                    # After finishing iteration over dependencies, remove old dependencies
                    for dependency_to_remove in dependencies_to_be_removed:
                        node.dependencies.remove(dependency_to_remove)

                    # Now add the new dependencies that we collected
                    node.dependencies.update(new_dependencies)

                    nodes_to_move.append(node)  # Appending the node to be moved

                # Move nodes to the upper level (merge them into the next level)
                for node in nodes_to_move:
                    next_level.append(node)
                    levels_costs[level_idx - 1] += node.cost

                # Remove nodes from the current level after moving
                for node in nodes_to_move:
                    current_level.remove(node)
                    levels_costs[level_idx] -= node.cost

                # Ensure that merged levels are correctly maintained
                if not current_level:  # If the current level becomes empty, shift the levels up
                    # We only need to shift levels if there are empty levels
                    if level_idx < len(levels_with_object) - 1:  # Check if there are levels to move
                        # Shift the next levels up
                        for i in range(level_idx, len(levels_with_object) - 1):
                            levels_with_object[i] = levels_with_object[i + 1]  # Move the next level up

                        # The last level is now empty, so we delete it
                        levels_with_object.pop()  # Remove the last level after shifting
                # Recalculate level costs after merging
               # calculate_level_costs()


            else:
                continue
        else:
            continue

    cost = 0
    temp_count = 1
    avg_cost = 0
    for i in levels_with_object:
        temp_count += 1
        for j in i:
            cost += j.cost
    avg_cost = cost / len(levels_with_object)
    output_widget.delete(1.0, tk.END)  # Clear previous output before displaying levels
    output_widget.insert(tk.END, f"Average cost:{avg_cost}\n")



                #burdan itibaren stage2, priority queue

    # Priority Queue oluşturma
    priority_levels = []  # Final priority-based levels array

    for level_idx, level in enumerate(levels_with_object):
        priority_queue = []  # Temporary heap for the current level
        for node in level:
            # Kendisine bağlı olan node'ların sayısını hesapla
            dependency_count = sum(1 for upper_level in levels_with_object[level_idx + 1:]
                                   for n in upper_level if node.index in n.dependencies)
            # Priority queue için tuple olarak ekle (öncelik değeri, node.index, node)
            heapq.heappush(priority_queue, (-dependency_count, node.index, node))

        # Priority queue'yu sıralı bir listeye çevir
        sorted_nodes = [heapq.heappop(priority_queue)[2] for _ in range(len(priority_queue))]
        priority_levels.append(sorted_nodes)  # Bu levelın sıralı nodelarını ekle







    most_dependent_nodes = get_most_dependent_nodes(priority_levels)  #get most dependent node indexes in levels

    # burdan sonra stage2: priority queue kullanarak taşıma işlemi








    update_dependant_from_lower()
    queue=update_queue(levels_with_object)

    for level_idx in range(len(levels_with_object)-1 , 1, -1):
        update_dependant_from_lower()
        queue = update_queue(levels_with_object)

        current_level = levels_with_object[level_idx]
        next_level = levels_with_object[level_idx - 1]

        nodes_to_move = []

        for node in current_level:
            # Update dependencies for the node being moved
            dependencies_to_be_removed = set()
            new_dependencies = set()  # Collect new dependencies to be added
            if queue[level_idx - 1] in node.dependencies:
                for dependant in next_level:
                    if dependant.index == queue[level_idx - 1]:
                        dependencies_to_be_removed.add(dependant.index)

                        for dependency in dependant.dependencies:
                            new_dependencies.add(dependency)

                        # After finishing iteration over dependencies, remove old dependencies
                        for dependency_to_remove in dependencies_to_be_removed:
                            node.dependencies.remove(dependency_to_remove)

                        # Now add the new dependencies that we collected
                        node.dependencies.update(new_dependencies)

                        nodes_to_move.append(node)  # Appending the node to be moved
                        node.move = True

        # Move nodes to the upper level
        for node in nodes_to_move:
            next_level.append(node)
            levels_costs[level_idx - 1] += node.cost

        temp_level = []

        for node in current_level:
            if node.move:
                continue
            else:
                temp_level.append(node)

        levels_with_object[level_idx] = temp_level

        for node in next_level:
            node.move = False

        for node in current_level:
            node.move = False

    for level_idx in range(len(levels_with_object) - 1, 1, -1):
        current_level = levels_with_object[level_idx]  # Current level in iteration
        if not current_level:
            levels_with_object.pop()
        else:
            break

    end_time = time.time()
    elapsed_time = end_time - start_time



    for level_idx in range(len(levels_with_object) - 1, 1, -1):

        current_level = levels_with_object[level_idx]  # Current level in iteration
        # Ensure that merged levels are correctly maintained
        # Ensure that merged levels are correctly maintained
        if not current_level:  # If the current level becomes empty, shift the levels up
            # We only need to shift levels if there are empty levels
            if level_idx < len(levels_with_object) - 1:  # Check if there are levels to move
                # Shift the next levels up
                for i in range(level_idx, len(levels_with_object) - 1):
                    levels_with_object[i] = levels_with_object[i + 1]  # Move the next level up

                # The last level is now empty, so we delete it
                levels_with_object.pop()  # Remove the last level after shifting

    final_level_size = len(levels_with_object)


    #  print the final result
    temp_increment_for_level = 0
    for level in levels_with_object:
        temp_level = []
        # Her seviyedeki node'ları al ve index'lerini ekle
        for node in level:
            temp_level.append(node.index)

        # Burada her seviyeyi düzgün yazdırıyoruz
        output_widget.insert(tk.END, f"Level {temp_increment_for_level}: {temp_level}\n")
        temp_increment_for_level += 1

    output_widget.insert(tk.END, f"Starting level size :{start_level_size}, Final level size:{final_level_size}\n")
    output_widget.insert(tk.END, f"Level size reduced by :{start_level_size - final_level_size} levels, which is by %{100-(final_level_size/start_level_size)*100}\n")

    output_widget.insert(tk.END, f"Elapsed Time: {elapsed_time}\n")


def update_dependant_from_lower():
    global levels_with_object

    list = levels_with_object

    for level in range(len(list) -1,1,-1):
        current_level = list[level]
        next_level = list[level-1]

        for node in next_level:
            node.dependant_from_lower = 0
        for node in next_level:
            for lower_node in current_level:
                if node.index in lower_node.dependencies:
                    node.dependant_from_lower+=1

def update_queue(queue):
    new_queue = []
    for level in range(0,len(queue)-1):
        temp_check = 0
        temp_index = 0
        current_level = queue[level]

        for node in current_level:
            if node.dependant_from_lower > temp_check:
                temp_check = node.dependant_from_lower
                temp_index = node.index

        new_queue.append(temp_index)

    return new_queue























def get_most_dependent_nodes(priority_levels):
    most_dependent_nodes = []
    for level in priority_levels:
        if level:
            first_index = level[0].index
            most_dependent_nodes.append(first_index)

    return most_dependent_nodes


def adjust_levels(levels_with_object):
    global levels_costs  #cost of the levels in the array format
    global average_cost  #calculated avg cost
   # global levels_with_object


    with open("Rewrite_list.txt", "w") as file:
        file.write("Rewriting list")

    for level_idx in range(len(levels_with_object)-1,1):
        current_level = levels_with_object[level_idx]
        next_level = levels_with_object[level_idx - 1]
        # list of nodes to move up from the current level to the previous level
        nodes_to_move_up = []



        #average cost'un %20sinden düşük costtaki levelları mergeleme
        if levels_costs[level_idx] < average_cost*0.45:   #%45
            if levels_costs[level_idx] + levels_costs[level_idx - 1] < average_cost*1.3:
                # if levels_costs[level_idx] + levels_costs[level_idx-1] < levels_costs[level_idx -1] + levels_costs[level_idx -2]:    ara level check

                    # MOVE THE NODES UP  -- MERGİNG
                for node in current_level:
                    with open("Rewrite_list.txt", "a") as file:
                        file.write("The node "+str(node.index)+"rewritten from level: "+str(current_level)+" to level:" +str(next_level)+"\n")


                    #dependencyleri update etmek için
                    for dependency in node.dependencies:
                        for dependant in levels_with_object[level_idx -1]:
                            if dependency == dependant.index:
                                node.dependencies.remove(dependency)
                                for dependant_dependency in dependant.dependency:
                                    node.dependencies.add(dependant_dependency)

                    #nodeları üst level'a taşımak için
                    next_level.append(node)
                    current_level.remove(node)
                    levels_costs[level_idx - 1] += node.cost
                    levels_costs[level_idx] -= node.cost

                for level_idtemp in range(current_level,len(levels_with_object)-2):
                    temp_current_level = levels_with_object[level_idtemp]
                    next_level = levels_with_object[level_idtemp +1]

                    temp_current_level = []
                    for node in next_level:
                        temp_current_level.append(node)
                    next_level= []

        ###burdan itibaren merge state devamı olarak kabul edilir


        #list of nodes to move up from the current level to the previous level
        nodes_to_move_up = []

        for node in current_level:
            dependency_in_upper=0
            for dependency in node.dependencies:
                if dependency in [n.index for n in next_level]:
                    dependency_in_upper+=1
            if dependency_in_upper == 1 and levels_costs[level_idx -1] < average_cost*120/100:
                nodes_to_move_up.append(node)



        #MOVE THE NODES UP
        for node in nodes_to_move_up:
            current_level.remove(node)
            levels_costs[level_idx-1] += node.cost
            levels_costs[level_idx] -= node.cost
            next_level.append(node)




    return levels_with_object




def read_sparse_matrix_csr(file_path):
    with open(file_path, 'r') as file:
        nrows, ncols, nnz = None, None, None
        for line in file:
            if line.startswith('%'):
                continue
            else:
                parts = line.strip().split()
                nrows = int(parts[0])
                ncols = int(parts[1])
                nnz = int(float(parts[2]))
                break

        if nrows is None or ncols is None or nnz is None:
            raise ValueError("Invalid matrix header")

        rows = []
        cols = []
        data = []

        for line in file:
            if not line.startswith('%'):
                row, col, value = map(float, line.strip().split())
                rows.append(int(row) - 1)
                cols.append(int(col) - 1)
                data.append(value)

        csr_matrix_data = sp.csr_matrix((data, (rows, cols)), shape=(nrows, ncols))
        return csr_matrix_data


def solve_sparse_system_with_lu(csr_matrix, b):
    csc_matrix = csr_matrix.tocsc()
    lu = splinalg.splu(csc_matrix)
    x = lu.solve(b)
    return x


def build_nodes(csr_matrix):
    global node_array
    nodes = [Node(i) for i in range(csr_matrix.shape[0])]
    temp_nnz = 0 #for nonzero count

    for i in range(csr_matrix.shape[0]):
        temp_nnz = 0
        for j in range(csr_matrix.indptr[i], csr_matrix.indptr[i + 1]):
            temp_nnz += 1
            if csr_matrix.indices[j] > i:
                continue
            if csr_matrix.indices[j] != i:
                nodes[i].dependencies.add(csr_matrix.indices[j])
                nodes[csr_matrix.indices[j]].dependents.append(i)

            nodes[i].values.append(csr_matrix.data[j])
            nodes[i].valueindexes.append(csr_matrix.indices[j])

        nodes[i].cost +=(2*temp_nnz)-1
    for node in nodes:
        cost = 2*(len(node.values)) - 1
        node.cost = cost

    node_array = nodes
    return nodes


# Global variables to store the selected file path and built nodes
selected_file_path = None
nodes = None


# New function to build nodes based on the selected file
def build_and_display_nodes(output_widget):
    global selected_file_path, nodes
    if selected_file_path:
        if selected_file_path.endswith(".mtx"):
            csr_matrix_data = read_sparse_matrix_csr(selected_file_path)
            nodes = build_nodes(csr_matrix_data)

            # Clear previous output
            output_widget.delete(1.0, tk.END)

            # Display the nodes
            output_widget.insert(tk.END, "Nodes:\n")
            for node in nodes:
                output_widget.insert(tk.END, f"{node}\n")

        elif selected_file_path.endswith(".tar.gz"):
            mtx_files = extract_mtx_files_from_tar_gz(selected_file_path)
            if mtx_files:
                csr_matrix_data = read_sparse_matrix_csr(mtx_files[0])  # Only the first .mtx file is used
                nodes = build_nodes(csr_matrix_data)

                # Clear previous output
                output_widget.delete(1.0, tk.END)

                # Display the nodes
                output_widget.insert(tk.END, "Nodes:\n")
                for node in nodes:
                    output_widget.insert(tk.END, f"{node}\n")
            else:
                messagebox.showerror("Error", "No .mtx file found in the .tar.gz archive.")
        else:
            messagebox.showerror("Error", "Please select a valid .mtx or .tar.gz file.")
    else:
        messagebox.showerror("Error", "No file selected. Please select a file first.")


# New function to analyze and display levels
def build_and_display_levels(output_widget):

    global nodes
    global levels
    global levels_costs
    global average_cost

    if nodes:
        # Clear previous output before displaying levels
        output_widget.delete(1.0, tk.END)

        levels = node_analysis(nodes, len(nodes))
        calculate_level_costs()
        level_cost=levels_costs

        # Display the levels
        output_widget.insert(tk.END, "Levels:\n")
        level_number = 0
        for level in levels:
            output_widget.insert(tk.END, f"Level {level_number}: {level}\n")
            level_number += 1
        output_widget.insert(tk.END, f"\nLevel cost: {level_cost}")
        output_widget.insert(tk.END, f"\nAverage cost: {average_cost}")
    else:
        messagebox.showerror("Error", "No nodes built. Please build nodes first.")


# Tkinter-based GUI
def open_file_dialog(output_widget):
    global selected_file_path
    selected_file_path = filedialog.askopenfilename(
        filetypes=[("MTX and TAR.GZ files", "*.mtx *.tar.gz"),
                   ("Matrix Files", "*.mtx"),
                   ("Compressed TAR files", "*.tar.gz"),
                   ("All Files", ".")]
    )
    if selected_file_path:
        output_widget.delete(1.0, tk.END)
        output_widget.insert(tk.END, f"Selected file: {selected_file_path}\n")


# Create main window

def clean_level_costs():
    global levels_costs

    temp_costs = []

    for i in levels_costs:
        if i == 0:
            continue
        else:
            temp_costs.append(i)
    levels_costs[:] = temp_costs

def calculate_level_costs():

    #levels_with_object arrayindeki objelere bakarak levelların costlarını int halinde levels_costs arrayine sırayla yazdırıyor
    global levels_with_object
    global levels
    global levels_costs
    levels_costs = []




    for level in levels_with_object:
        level_cost = 0
        node_counter = 0
        #level_cost değişkeninde nonzero sayısı
        for node in level:
            node_counter+=1
            level_cost+=len(node.values)
        levels_costs.append(2*(level_cost) - node_counter)

    calculate_average_level_cost()





def calculate_average_level_cost():
    global average_cost
    global levels_costs
    average_cost = 0


    total_cost = 0
    for i in levels_costs:
        total_cost+= i
    average_cost = total_cost/len(levels_costs)




def create_gui():
    root = tk.Tk()
    root.title("Sparse Matrix Solver")

    # Create and configure widgets
    frame = tk.Frame(root)
    frame.pack(padx=20, pady=20, expand=True, fill='both')  # Frame'in büyümesine izin ver

    label = tk.Label(frame, text="Please select a matrix file (.mtx or .tar.gz):")
    label.pack(pady=10)

    open_button = tk.Button(frame, text="Open File", command=lambda: open_file_dialog(output_widget))
    open_button.pack(pady=5)

    build_nodes_button = tk.Button(frame, text="Build Nodes", command=lambda: build_and_display_nodes(output_widget))
    build_nodes_button.pack(pady=5)

    build_levels_button = tk.Button(frame, text="Build Levels", command=lambda: build_and_display_levels(output_widget))
    build_levels_button.pack(pady=5)

    build_levels_button = tk.Button(frame, text="Build Rewriting list", command=lambda: deneme(output_widget))
    build_levels_button.pack(pady=5)

    build_levels_button = tk.Button(frame, text="Analyze and Build Cache", command=lambda: unoptimized_deneme(output_widget))
    build_levels_button.pack(pady=5)

    # Create Scrollbar
    scroll_bar = tk.Scrollbar(frame)
    scroll_bar.pack(side=tk.RIGHT, fill=tk.Y)

    output_widget = tk.Text(frame, wrap=tk.WORD, height=20, width=80, yscrollcommand=scroll_bar.set)
    output_widget.pack(expand=True, fill='both', pady=10)  # Text widget'in genişlemesini sağla

    # Configure scrollbar
    scroll_bar.config(command=output_widget.yview)

    root.mainloop()


if __name__ == "__main__":
    create_gui()
