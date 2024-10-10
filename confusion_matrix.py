def confusion_matrix(file_name):
   with open(file_name, 'r') as file:
        lines = file.readlines()
        confusion_matrix = []

        for line in lines:
            line = line.rstrip("\n") #getting rid of newline character
            nums_as_strings = line.split(" ")

            row = [int(i) for i in nums_as_strings]

            confusion_matrix.append(row)
   return confusion_matrix


if __name__ == "__main__":
    confusion_matrix_nb = confusion_matrix("confusion_matrix_nb.txt")
    confusion_matrix_ten_fold = confusion_matrix("confusion_matrix_ten_fold.txt")
    confusion_matrix_k_star = confusion_matrix("confusion_matrix_k_star.txt")

