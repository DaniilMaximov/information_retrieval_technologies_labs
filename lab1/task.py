import csv
import matplotlib.pyplot as plt

class ReportProcessor:
    def __init__(self, input_csv='report.csv', output_csv='report_filtered.csv'):
        self.input_csv = input_csv
        self.output_csv = output_csv
        self.max_positions = 0
        self.count_with_degree = 0
        self.unique_positions = set()
        self.positions_distribution = []

    def process_report(self):
        with open(self.input_csv, 'r', encoding='utf-8') as infile, \
             open(self.output_csv, 'w', newline='', encoding='utf-8') as outfile:

            reader = csv.DictReader(infile)
            writer = csv.DictWriter(outfile, fieldnames=reader.fieldnames)
            writer.writeheader()

            for row in reader:
                num_positions = int(row['Количество должностей'])
                academic_degree = row['Учёная степень'] == 'True'

                self.positions_distribution.append(num_positions)

                if num_positions > self.max_positions:
                    self.max_positions = num_positions

                if academic_degree:
                    self.count_with_degree += 1

                self.update_unique_positions(row)

                if num_positions >= 2:
                    writer.writerow(row)

        self.display_results()
        self.plot_positions_distribution()

    def update_unique_positions(self, row):
        positions = row['Количество должностей']
        if positions:
            self.unique_positions.update(positions.split(','))

    def display_results(self):
        print(f'Максимальное число должностей у сотрудника: {self.max_positions}')
        print(f'Количество сотрудников с учёной степенью: {self.count_with_degree}')

    def get_unique_positions(self):
        return sorted(self.unique_positions)

    def plot_positions_distribution(self):
        plt.hist(self.positions_distribution, bins=range(1, self.max_positions + 2), edgecolor='black')
        plt.xlabel('Количество должностей')
        plt.ylabel('Количество сотрудников')
        plt.title('Распределение количества должностей среди сотрудников')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.show()


class UniquePositionsWriter:
    def __init__(self, unique_positions, output_file='unique_positions.txt'):
        self.unique_positions = unique_positions
        self.output_file = output_file

    def write_unique_positions(self):
        with open(self.output_file, 'w', encoding='utf-8') as f:
            for position in self.unique_positions:
                f.write(position + '\n')
        print(f'Количество уникальных должностей: {len(self.unique_positions)}')


if __name__ == '__main__':
    report_processor = ReportProcessor()
    report_processor.process_report()

    # unique_positions_writer = UniquePositionsWriter(report_processor.get_unique_positions())
    # unique_positions_writer.write_unique_positions()
