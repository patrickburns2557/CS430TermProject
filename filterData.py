#Filter out all data entries that have unknown values since the original paper did that as well

output = open("adult.filtered", "w")

with open("adult.data") as input:
    for line in input:
        if "?" not in line:
            output.write(line)
