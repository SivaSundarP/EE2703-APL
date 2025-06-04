import numpy as np


def evalSpice(filename):
    """Function which takes in the .ckt file and returns the
    solution for the circuit network described in the .ckt file.
    The task is split into parts which are executed by different functions:
        -> comp_extract      :extracts the circuit details from .ckt file
                              and stores them in a dictionary.
        -> initialize_VI     :extracts node details from the components dictionary
                              and initializes two dictionaries (V and I) which takes care of nodes
                              and currents via voltage source
        -> matrix_initialize :initializes the MNA matrices with size of
                              square matrix M and column matrix N being
                              "no. of unknown nodes" + "no. of voltage sources".
        -> VIR_split         :splits the component dictionary into three lists V_src, I_src and R. They
                              contain details about voltage sources, current sources and resistors resp.
        -> matrix_build      :builds the MNA matrices M and N, using V_src, I_src, R and V.
    """

    # Dictionaries to store V and I values
    V = {}
    I = {}  # Contains only the currents through voltage sources

    # Extracting details of components in ckt file and storing in a dictionary
    comp = comp_extract(filename)

    # Initializing I with voltage source names as keys
    initialize_VI(comp, V, I)

    # Initializing the matrix
    M = []  # For network
    N = []  # For inputs
    size = len(V) + len(I) - 1  # Size of matrix (Excluding GND)
    matrix_initialize(M, N, size)

    # Sorting the components into three groups
    V_src, I_src, R = VIR_split(comp)

    # Building the matrix
    matrix_build(V_src, I_src, R, M, N, V)

    try:
        X = np.linalg.solve(M, N)
    except np.linalg.LinAlgError:
        raise ValueError("Circuit error: no solution")

    # Writing the solution from X to V and I dicts
    index = 0  # For traversing the X vector
    del V["GND"]  # Removing it for now
    for key in V:
        V[key] = X[index]
        index += 1
    V["GND"] = 0  # Adding "GND"
    for key in I:
        I[key] = X[index]
        index += 1

    return (V, I)


def comp_extract(ckt_file):
    """Function to extract the circuit details from a .ckt file.
    The function assumes SPICE circuit structure (Standard format),
    and also it assumes NO FLOATING NODES.

    Returns a dictionary whose keys are element names and
    values are list which contain details about the element.
    Raises errors as and when needed."""

    # Temp. dictionary to store component details
    comp = {}

    # Opening the ckt file in read mode
    try:
        with open(ckt_file, "r") as file:
            lines = file.readlines()  # Taking the lines

            # Extracting lines between .circuit and .end
            start = -1  # Initialized with -1 to represent "not updated"
            stop = -1
            for i in range(len(lines)):
                if ".circuit" in lines[i]:
                    start = i
                if ".end" in lines[i]:
                    stop = i
            lines = lines[start + 1 : stop]
    except FileNotFoundError:
        raise FileNotFoundError("Please give the name of a valid SPICE file as input")

    # Checking File errors
    if start == -1 or stop == -1 or stop <= start:
        raise ValueError("Malformed circuit file")

    for line in lines:  # Iterating through the lines
        words = line.split()  # Spliting the line into words

        if (
            words == [] or words[0].startswith("#") or words[0].startswith("#")
        ):  # Neglect empty lines
            continue
        if words[0][0] == "V" or words[0][0] == "I":
            # Error Handling
            if (
                len(words) < 5
            ):  # Words should contain 5 elements for V and I source, else return -1
                raise ValueError(f"'{words[0]}' should contain 4 data")
            if not is_float(
                words[4]
            ):  # Words should contain 4 elements for resistors, else return -1
                raise ValueError(f"{words[0]} contain non-numeric source value")
            if words[0] in comp:  # If an element is defined again
                raise ValueError(f"{words[0]} defined multiple times in the circuit")

            comp[words[0]] = [  # for voltage and current source, there are 4 words
                words[1],
                words[2],
                words[3],
                words[4],
            ]

        elif words[0][0] == "R":
            # Error Handling
            if (
                len(words) < 4
            ):  # Words should contain 4 elements for resistors, else return -1
                raise ValueError(f"'{words[0]}' should contain 3 data")
            if not is_float(words[3]):
                raise ValueError(f"{words[0]} contain non-numeric resistance value")
            if words[0] in comp:  # If an element is defined again
                raise ValueError(f"{words[0]} defined multiple times in the circuit")

            comp[
                words[0]
            ] = [  # words = [component name, terminal 1, terminal 2, value]
                words[1],
                words[2],
                words[3],
            ]

        else:  # Invalid element
            raise ValueError("Only V, I, R elements are permitted")

    # Checking whether there are any floating nodes in the circuit
    if is_floating_node(comp):
        raise ValueError(
            f"There are floating nodes in the circuit described in '{ckt_file}' file"
        )

    return comp


def is_float(s):
    """Function to check whether the string can be converted to float or not.
    eg: is_float("1e3") == True
        is_float("abc.ef") == False"""
    try:
        float(s)  # Try to convert the string to a float
        return True

    except ValueError:
        return False


def initialize_VI(comp, V, I):
    """Function takes in the component dictionary and two other dictionaries (V, I)
    which gets modified by the function such that:
        V: represents node voltages, keys as nodes and
           values as node potentials (initialized to 0)
        I: represents the currents via ext. voltage sources,
           keys as voltage source names and values as
           the current value (intialized to 0)
    The function takes the nodes values of each element in "comp" and stores them in V as keys
    For I, it takes the names of voltage sources in "comp"."""

    # Temporary dictionaries for getting the keys
    V_temp = {}
    I_temp = {}

    for key in comp:
        if key[0] == "V":
            I_temp[key] = 0  # Track voltage sources in I dictionary

        # Terminals of the component give nodes
        node_1 = comp[key][0]
        node_2 = comp[key][1]

        # Add nodes to V_temp
        V_temp[node_1] = 0
        V_temp[node_2] = 0

    """ Sorting is done because the MNA matrices are created in a particular
        order (see docs of matrix_build function) which results in the 
        solution matrix X having its elements correspond to nodes
        (and then currents) which follow ascending order.
            ie, X[n] corresponds to (n+1)th node (OR current via ([n+1]-N)th voltage source,
                                                  where N is the total number of nodes) """
    if "GND" not in V_temp:
        raise ValueError("Circuit must have a GND node")

    V_temp["GND"] = 0  # Start with GND as 0

    # Sort the keys of V_temp and I_temp
    sorted_keys_v = sorted(V_temp.keys())
    sorted_keys_i = sorted(I_temp.keys())

    # V and I dictionary based on the sorted order
    for key in sorted_keys_v:
        V[key] = 0  # Initialize the values in V to 0
    for key in sorted_keys_i:
        I[key] = 0  # Initialize the values in V to 0


def matrix_initialize(M, N, size):
    """Function to initialize the lists M and N which represent
    the matrices on the LHS and RHS of MNA matrix eqn:
        M*X = N, where M represents the network
                       N represent the external sources
                       X contain node voltages and currents through ext. voltage sources
    """
    for i in range(size):
        N.append(0)
        M.append([])
        for j in range(size):
            M[i].append(0)


def VIR_split(comp):
    """Function which takes the component dictionary and splits it into three lists and returns them:
        V: contains the voltage sources
        I: contains the current sources
        R: contains the resistors
    This splitting is done for making the MNA matrices with ease."""

    # Initializing lists
    V_src = []
    I_src = []
    R = []

    # Iterating through the components dictionary
    for key in comp:
        element = comp[key]
        if key[0] == "V":  # Voltage source
            element.append(key)
            V_src.append(element)
        elif key[0] == "I":  # Current source
            element.append(key)
            I_src.append(element)
        elif key[0] == "R":  # Resistor
            element.append(key)
            R.append(element)

    return V_src, I_src, R


def matrix_build(V_src, I_src, R, M, N, V):
    """Function used for building the MNA matrices M and N
    using the list of voltage sources, current sources,
    resistors and dictionary with node names.

    KCL: Sum of currents leaving a node is zero
         Mathematically, I1+I2+I3+....=0
         Usage: Sum of currents leaving a node   =  Sum of currents entering the node through
               via resistor and voltage branches    current source branches entering the node

    KVL: Sum of voltage diff. across a loop is zero
         Usage: Voltage difference across   =  Voltage difference produced
               the nodes of voltage source       by the voltage source

    MNA equation: M*X = N
        where,
            M contains the LHS of KCL and KVL equations
            N contains the RHS of KCL and KVL equations
            X contains the node voltages and currents via voltage source.

    Structure of M and N matrices:
    Consider a network with resistors, voltage sources and current sources connected in
    some arbitrary way. Say it has k nodes. To solve this network, we follow the procedure:

    -> Identify the nodes of the circuit (which are stored in V).

    -> Write KCL equations assuming currents in voltage sources
       (say I["V1"] => current via "V1" voltage source)

       LHS is stored in M whose row number corresponds to the
       node number whose KCL equation is stored in that row,
       and each column number corresponds to the variable number:
            column 0 => V1, column 1 => V2...column k-1 => Vk
            column k => I[V1], column k+1 => I[V2]....

       RHS is stored in N as follows:
            N[0] => RHS of KCL for node 1,
            N[k-1] => RHS of KCL for node k...

    -> After the k equations for k nodes (Note: when im saying nodes,
       I mean the unsolved nodes, hence, neglecting GND node whose
       value is fixed as 0), next comes KVL equations.

       LHS is stored in M as follows:
            row 0 => KCL (n1), row 1 => KCL (n2)...row k-1 => KCL (nk)
            row k => KVL (V1), row k+1 => KVL (V2)....

       RHS is stored in N as follows:
            N[k] => RHS of KVL for source V1,
            N[k+1] => RHS of KVL for source V2...

    Hence, the solution X, will have the order:
        X[0:k-1] = [n1, n2,...,nk]
        X[k:k+m] = [I["V1"], I["V2"],..., I["Vm"]]  where, m is the number of voltage sources
    """

    # The KCL equations come first
    # Writing the contributions by resistors first
    for node in V:  # Each node has its equation (row_no is node_no)
        if node == "GND":
            continue
        for (
            resistor
        ) in (
            R
        ):  # Each resistor stores its values in the columns according to its terminals
            i = extract_node_num(resistor[0]) - 1 if resistor[0] != "GND" else -1
            j = extract_node_num(resistor[1]) - 1 if resistor[1] != "GND" else -1
            eqn_no = extract_node_num(node) - 1
            if (
                node == resistor[0] or node == resistor[1]
            ):  # If resistor connected to the node...
                if extract_node_num(node) == i + 1:
                    M[eqn_no][i] += 1 / float(resistor[-2])
                    M[eqn_no][j] -= 1 / float(resistor[-2])
                else:
                    M[eqn_no][j] += 1 / float(resistor[-2])
                    M[eqn_no][i] -= 1 / float(resistor[-2])

    # Now writing the contributions by current sources
    for src in I_src:
        exit_node = extract_node_num(src[0]) - 1 if src[0] != "GND" else -1
        entry_node = extract_node_num(src[1]) - 1 if src[1] != "GND" else -1
        N[exit_node] -= float(src[-2])
        N[entry_node] = float(src[-2])

    # Lastly, writing the contributions by the currents through voltage source
    for src in V_src:
        pos_node = extract_node_num(src[0]) - 1 if src[0] != "GND" else -1
        neg_node = extract_node_num(src[1]) - 1 if src[1] != "GND" else len(M) - 1
        source_num = (
            extract_node_num(src[-1]) if extract_node_num(src[-1]) != None else 1
        )
        total_node = len(V) - 1 - 1
        for clr_node in range(0, len(M)):
            M[clr_node][total_node + source_num] = 0
        M[pos_node][total_node + source_num] = 1 if pos_node != -1 else 0
        M[neg_node][total_node + source_num] = -1 if neg_node != len(M) - 1 else 0

    # Now equations from voltage sources
    eqn_no = (
        len(V) - 1
    )  # Excluding GND, and this also gives the eqn_no in matrix as well
    for src in V_src:
        pos_node = extract_node_num(src[0]) - 1 if src[0] != "GND" else -1
        neg_node = extract_node_num(src[1]) - 1 if src[1] != "GND" else -1
        source_num = (
            extract_node_num(src[-1]) if extract_node_num(src[-1]) != None else 1
        )
        if pos_node != -1:
            M[eqn_no][pos_node] = 1
        if neg_node != -1:
            M[eqn_no][neg_node] = -1
        N[eqn_no] = float(src[-2])
        eqn_no += 1


def is_floating_node(comp):
    """Function which checks whether the circuit contains floating
    nodes (nodes which are not connected to another node). For a
    circuit without floating nodes, each node should be specified at
    least twice. If this condition fails, then there exists floating nodes."""

    # Lists for storing Unique and Duplicate nodes for checking
    # whether there are floating nodes in the given circuit (these are invalid circuits)
    U = []
    D = []

    for key in comp:
        element = comp[key]

        # Terminals of the element
        node_1 = extract_node_num(element[0])
        node_2 = extract_node_num(element[1])

        if node_1 not in U:
            U.append(node_1)
        else:
            D.append(node_1)

        if node_2 not in U:
            U.append(node_2)
        else:
            D.append(node_2)

    if set(U) == set(D):
        return False
    else:
        return True


def extract_node_num(s):
    """Function used to extract the node numbers from node names.
    Given, a string of the form "str with no number"+"number"+"string with no number"
    the function returns the number as integer."""
    number = ""
    for char in s:
        if char.isdigit():
            number += char
    return int(number) if number else None
