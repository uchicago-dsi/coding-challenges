"""
Make a csv with all zero data to test the ambiguous case between top
and bottom. With no evidence, there should be no rationale between
choosing one or the other.
"""

def make_null_dataset() -> None:
    fname = "./train/top.csv"
    with open(fname, 'r') as f:
        lines = f.readlines()
    # Keep the header line
    output = lines[0]
    for line in lines[1:]:
        vals = line.split(',')
        # Keep the first column
        output += f"{vals[0]},"
        # Zero all data
        zeroes = ["0" for _ in vals[1:]]
        output += ','.join(zeroes)

    with open("./train/null.csv", 'w') as f:
        f.write(output)


if __name__ == "__main__":
    make_null_dataset()
