import csv


def save_data_to_csv(users):
    with open('csv_files/task.csv', mode="r", newline="", encoding="utf-8") as infile, \
            open('csv_files/submission.csv', mode="w", newline="", encoding="utf-8") as outfile:
        reader = csv.reader(infile, delimiter=";")
        writer = csv.writer(outfile, delimiter=";")

        for row in reader:
            user_id = int(row[1])
            movie_id = int(row[2])
            rating = row[3]

            for user in users:
                for movie in user.test_set:

                    if rating == "NULL" and user_id == user.id and movie_id == movie.id:
                        row[3] = movie.rating
                        writer.writerow(row)
                        break

    print(f"Updated file saved...")