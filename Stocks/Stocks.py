#!/usr/bin/env python
"""
stocks.py
Template by: Gary Lee (6.008 TA, Fall 2021)

Please read the project instructions beforehand! Your code should go in the
blocks denoted by "YOUR CODE GOES HERE" -- you should not need to modify any
other code!
"""

# import packages here
import numpy as np
import matplotlib.pyplot as plt
import time

# Information for Stocks A and B
priceA = np.loadtxt('data/priceA.csv')
priceB = np.loadtxt('data/priceB.csv')



# DO NOT RENAME OR REDEFINE THIS FUNCTION.
# THE compute_average_value_investments FUNCTION EXPECTS NO ARGUMENTS, AND
# SHOULD OUTPUT 2 FLOAT VALUES
def compute_average_value_investments():
    # -------------------------------------------------------------------------
    # YOUR CODE GOES HERE FOR PART (b)
    patterns = [np.binary_repr(i, width=20) for i in range(2 ** 20)]
    total_hold = []
    total_rebalance = [1]
    for k in patterns:
        total = 0
        to_multiply = 1
        val_c = [.5]
        val_d = [.5]
        for z in range(len(k)):
            val_c.append(val_c[z] * 1.05)
            if int((k[z])) == 0:
                to_multiply *= 1.4
                val_d.append(val_d[z] * 1.4)
            else:
                to_multiply *= .7875
                val_d.append((val_d[z] * .7875))
            total = (.5 * to_multiply) + (.5 * (1.05 ** 20))
            # print(val_c)
            rebalance = (val_d[z + 1] + val_c[z + 1]) / 2
            val_c[z + 1] = rebalance
            val_d[z + 1] = rebalance
        total_hold.append(total)
        total_rebalance.append(rebalance * 2)
    average_buyandhold = sum(total_hold) / (2 ** 20)
    average_rebalancing = sum(total_rebalance) / (2 ** 20)
    return average_buyandhold, average_rebalancing
    # ASSIGN YOUR FINAL VALUES TO THE RESPECTIVE VARIABLES, I.E.,
    # average_buyandhold = (YOUR ANSWER FOR BUY & HOLD)
    # average_rebalancing = (YOUR ANSWER FOR REBALANCING)
    #
    # END OF YOUR CODE FOR PART (b)
    # -------------------------------------------------------------------------


def part_c(n):
    patterns = [np.binary_repr(i, width=n) for i in range(2 ** n)]
    buy_and_hold = 0
    for k in patterns:
        total = 0
        to_multiply = 1
        val_c = [.5]
        val_d = [.5]
        for z in range(len(k)):
            val_c.append(val_c[z] * 1.05)
            if int((k[z])) == 0:
                to_multiply *= 1.4
                val_d.append(val_d[z] * 1.4)
            else:
                to_multiply *= .7875
                val_d.append((val_d[z] * .7875))
            total = (.5 * to_multiply) + (.5 * (1.05 ** n))
            rebalance = (val_d[z + 1] + val_c[z + 1]) / 2
            val_c[z + 1] = rebalance
            val_d[z + 1] = rebalance
        if total > rebalance * 2:
            buy_and_hold += 1
    return "Buy and Hold was better " + str(buy_and_hold) + "/" + str(2**n) + " times"


# DO NOT RENAME OR REDEFINE THIS FUNCTION.
# THE compute_doubling_rate_investments FUNCTION EXPECTS NO ARGUMENTS, AND
# SHOULD OUTPUT 2 FLOAT VALUES
def compute_doubling_rate_investments():
    # -------------------------------------------------------------------------
    # YOUR CODE GOES HERE FOR PART (e)
    #
    # ASSIGN YOUR FINAL VALUES TO THE RESPECTIVE VARIABLES, I.E.,
    # doubling_rate_buyandhold = (YOUR ANSWER FOR BUY & HOLD)
    # doubling_rate_rebalancing = (YOUR ANSWER FOR REBALANCING)
    total_patterns = [np.binary_repr(i, width=20) for i in range(2 ** 20)]
    even_patterns = []
    total_hold = []
    total_rebalance = [1]
    for w in total_patterns:
        even = True
        num_zeros = 0
        for u in range(len(w)):
            if int((w[u])) == 0:
                num_zeros += 1
        if num_zeros == 10:
            even_patterns.append(w)
    for k in even_patterns:
        total = 0
        to_multiply = 1
        val_c = [.5]
        val_d = [.5]
        for z in range(len(k)):
            val_c.append(val_c[z] * 1.05)
            if int((k[z])) == 0:
                to_multiply *= 1.4
                val_d.append(val_d[z] * 1.4)
            else:
                to_multiply *= .7875
                val_d.append((val_d[z] * .7875))
            total = (.5 * to_multiply) + (.5 * (1.05 ** 20))
            rebalance = (val_d[z + 1] + val_c[z + 1]) / 2
            val_c[z + 1] = rebalance
            val_d[z + 1] = rebalance
        total_hold.append(total)
        total_rebalance.append(rebalance * 2)
    average_buyandhold = np.asarray(sum(total_hold) / (len(even_patterns)))
    average_rebalancing = np.asarray(sum(total_rebalance) / (len(even_patterns)))
    doubling_rate_buyandhold = float(1 / 20 * np.log2(average_buyandhold))
    doubling_rate_rebalancing = float(1 / 20 * np.log2(average_rebalancing))
    # END OF YOUR CODE FOR PART (e)
    # -------------------------------------------------------------------------
    return doubling_rate_buyandhold, doubling_rate_rebalancing

def part_a():
    val_A = np.array([.5])
    val_B = np.array([.5])
    val_A_1 = np.array([.5])
    val_B_1 = np.array([.5])
    total = np.array([1])
    total_1 = np.array([1])
    for i in range(0, len(priceA) - 1):
        percent_A = priceA[i + 1] / priceA[i]
        percent_B = priceB[i + 1] / priceB[i]
        val_A = np.append(val_A, val_A[i] * percent_A)
        val_B = np.append(val_B, val_B[i] * percent_B)
        equal_val = (val_A_1[i] * percent_A + val_B_1[i] * percent_B) / 2
        val_A_1 = np.append(val_A_1, equal_val)
        val_B_1 = np.append(val_B_1, equal_val)
        total = np.append(total, val_A[i + 1] + val_B[i+1])
        total_1 = np.append(total_1, val_A_1[i + 1] + val_B_1[i+1])
    plt.plot(total, label = "Buy and Hold")
    plt.plot(total_1, label = "Constant Redistribution")
    plt.title("Part A")
    plt.legend()
    plt.show()


def part_d():
    times = []
    for i in range(1, 21):
        start_time = time.time()
        part_c(i)
        times.append(time.time() - start_time)
    plt.yscale("log")
    plt.plot(times[4:len(times)])
    plt.show()

def main():

    # print("PART (b)")
    # average_buyandhold, average_rebalancing = compute_average_value_investments()
    # print(f'Computed Averaged Value for Buy & Hold: {average_buyandhold}')
    # print(f'Computed Averaged Value for Rebalancing: {average_rebalancing}')
    # print()
    #
    # print("PART (e)")
    # doubling_rate_buyandhold, doubling_rate_rebalancing = compute_doubling_rate_investments()
    # print(f'Computed Doubling Rate for Buy & Hold: {doubling_rate_buyandhold}')
    # print(f'Computed Doubling Rate for Rebalancing: {doubling_rate_rebalancing}')
    # print()

    # -------------------------------------------------------------------------
    # part_a()
    part_d()
    # -------------------------------------------------------------------------



if __name__ == '__main__':
    main()
