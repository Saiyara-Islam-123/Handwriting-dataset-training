def confusion_matrix(file_name):
   with open(file_name, 'r') as file:
        lines = file.readlines()
        cm = []

        for line in lines:
            line = line.rstrip("\n") #getting rid of newline character
            nums_as_strings = line.split(" ")

            row = [int(i) for i in nums_as_strings]

            cm.append(row)
   return cm


def most_confused_digit(cm, num):

    most_confused_digit_count = 0
    confused_digit = 999

    for i in range(len(cm[num])):
        if cm[num][i] > most_confused_digit_count and i != num:
            most_confused_digit_count = cm[num][i]
            confused_digit = i


    return confused_digit

def most_confused_digits_all_rows(cm):

    d = {}

    for i in range(0, 10):

        d[i] = most_confused_digit(cm, i)

    return d


if __name__ == "__main__":
    confusion_matrix_nb = confusion_matrix("confusion_matrix_nb.txt")
    confusion_matrix_ten_fold = confusion_matrix("confusion_matrix_ten_fold.txt")
    confusion_matrix_k_star = confusion_matrix("confusion_matrix_k_star.txt")

    print(most_confused_digits_all_rows(confusion_matrix_nb))
    print(most_confused_digits_all_rows(confusion_matrix_ten_fold))
    print(most_confused_digits_all_rows(confusion_matrix_k_star))