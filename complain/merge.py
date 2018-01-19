import pandas
import getopt
import sys



def parse_args(argv):
    firstfile = ''
    secondfile = ''
    outputfile = ''

    try:
        opts, args = getopt.getopt(argv, "hx:y:o:")
    except getopt.GetoptError:
        print('merge.py -x <firstfile> -y <secondfile> -o <outputfile>')
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            print('merge.py -x <firstfile> -y <secondfile> -o <outputfile>')
            sys.exit()
        elif opt == '-x':
            firstfile = arg
        elif opt == '-y':
            secondfile = arg
        elif opt == '-o':
            outputfile = arg
        else:
            print('merge.py -x <firstfile> -y <secondfile> -o <outputfile>')
            sys.exit()

    if not firstfile and not secondfile and not outputfile:
        print('merge.py -x <firstfile> -y <secondfile> -o <outputfile>')
        sys.exit(2)

    return firstfile, secondfile, outputfile

if __name__ == "__main__":
    first, second, output = parse_args(sys.argv[1:])

    first_df = pandas.read_csv(open(first, encoding='latin-1'), delimiter="|")
    second_df = pandas.read_csv(open(second, encoding='latin-1'), delimiter="|")

    frames = [first_df, second_df]
    result = pandas.concat(frames)
    result.to_csv(output, sep='|', encoding='latin-1')
    print("Done.")
