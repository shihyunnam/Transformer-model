def countpairs(projectcosts, target):
    count = 0
    newmap = {}
    for cost in projectcosts:
        newmap[cost] = cost - target 
    print(newmap)
        

print(countpairs([1,3,5], 2))
        

    