import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd
import openpyxl


class CyberAssistant:
    def __init__(self, df=None):
        self.df = df
        self.copy_df = df  # испольуется для хранения начальной информации
    def download(self, file_name, no_nan=True):
        """
        загружает данные из excel файла
        :param no_nan: нужно ли удалять nan-ы
        :param file_name: название файла
        """
        self.df = pd.read_excel(file_name, index_col=None)
        if no_nan:
            self.df = self.df.replace(' ', np.nan)
            self.df = self.df.dropna()

    def recalculation_1param(self, function, data_label, new_label):
        """
        производит пересчет величин по заданной функции, добавляет новую колонку в df, в пересчете участвует 1 колонка
        используется например для перевода в другие единици измерения и для смещения
        :param new_label: название пересчитанной величины
        :param function: функция пересчета
        :param data_label: наименование пересчитываемой величины
        """
        new_column = function(self.df[data_label].values)
        self.df[new_label] = new_column

    def recalculation_2param(self, function, data_label1, data_label2, new_label):
        """
        производит пересчет величин по заданной функции, добавляет новую колонку в df, в пересчете участвуют 2 колонки
        :param new_label: название пересчитанной величины
        :param function: функция пересчета
        :param data_label1: наименование 1 величины, участвующей в пересчете
        :param data_label2: наименование 2 величины, участвующей в пересчете
        """
        new_column = function(self.df[data_label1].values, self.df[data_label2].values)
        self.df[new_label] = new_column

    def graph(self, name, x_label, y_label, mode=True):
        """
        рисует график
        :param name: имя графика
        :param x_label: название оси абсцисс
        :param y_label: название оси ординат
        :param mode: по умолчанию =True, если =True, то аппроксимирует прямой, выводит коэффициенты и погрешности
        :return p, v: параметры аппроксимации
        """
        plt.scatter(self.df[x_label].values, self.df[y_label].values)
        plt.title(name)
        plt.ylabel(y_label)
        plt.xlabel(x_label)
        plt.grid()
        if mode:
            p, v = np.polyfit(self.df[x_label].values, self.df[y_label].values, deg=1, cov=True, full=False)
            poly = np.poly1d(p)
            left = float(min(self.df[x_label].values))
            right = float(max(self.df[x_label].values))
            x = np.linspace(left, right, 100)
            plt.plot(x, poly(x))
            print('$$k_{угл} = %f \pm %f $$' % (p[0], math.sqrt(v[0][0])))
            print('$$b = %f \pm %f $$' % (p[1], math.sqrt(v[1][1])))
            plt.show()
            return p, v
        plt.show()
        # plt.savefig(filename) # можно не рисовать, а сохранять

    def hist(self, cols):
        """
        рисует гистограмму
        :param cols: колонки, для которых необходима гистограмма
        """
        self.df.hist(column=cols, figsize=(10, 7))
        plt.show()

    def statistic_in_cols(self, cols, average_col_name, variance_col_name):
        """
        считает среднее и среднеквадратичное отклонение по колонкам и добавляет новую колонку в df
        применяется в ситуации, когда нужно произвести измерение одной величины несколько раз, найти среднее и
        среднекавдратичное отклонение
        ЕСЛИ СОБЫТИЕ СЛУЧАЙНОЕ, НУЖНО ДЕЛИТЬ ЕЩЕ НА КОРЕНЬ ИЗ N!!!
        :param variance_col_name: название колонки среднеквадратичного отклонения
        :param cols: названия колонок, по которым счииается среднее
        :param average_col_name: имя для новой колонки
        """
        av_column = np.zeros_like(self.df[cols[0]].values)
        for i in cols:
            av_column = av_column + self.df[i].values
        av_column = av_column/len(cols)
        self.df[average_col_name] = av_column
        var_column = np.zeros_like(self.df[cols[0]].values)
        for i in cols:
            var_column = var_column + (self.df[i].values - av_column)**2
        var_column = np.sqrt(var_column/(len(cols) - 1))
        self.df[variance_col_name] = var_column

    def statistic_in_one_col(self, col):
        """
        считает среднее значение и отклонение по колонке
        ЕСЛИ СОБЫТИЕ СЛУЧАЙНОЕ, НУЖНО ДЕЛИТЬ ЕЩЕ НА КОРЕНЬ ИЗ N!!!
        :param col: название колонки
        :return: среднее значение по колонке и среднеквадратичное отклонение
        """
        average = np.mean(self.df[col].values)
        variance = 0
        for i in self.df[col].values:
            variance = variance + (i - average) ** 2
        return average, np.sqrt(variance/(len(self.df[col]) - 1))

    def latex_table(self):
        """
        печатает латех табличку
        """
        print(self.df.to_latex())
        # если данных много, то здесь сделать слайс, подсмотреть можно в уроке по pandas

# если данных не много, то удобнее делать df по словарю
# some_dict = {'one': pd.Series([1,2,3], index=['a','b','c']),
#              'two': pd.Series([1,2,3,4], index=['a','b','c','d']),
#              'three': pd.Series([5,6,7,8], index=['a','b','c','d'])}
# df = pd.DataFrame(some_dict)


# John = CyberAssistant()
# John.download('Книга1.xlsx')
# print(John.df)

# John.recalculation_1param(lambda x: x*10, 'Напряжение', 'Напряжение_10')
# print(John.df)

# John.recalculation_2param(lambda x, y: x/y, 'Напряжение', 'Ток', 'Сопротивление')
# print(John.df)

# p, v = John.graph('Напряжение от тока', 'Ток', 'Напряжение')

# John.hist(['Ток', 'Напряжение'])

# John.statistic_in_cols(['Ток', 'Напряжение'], 'av', 'var')  # это бред по-сути, просто проверка работы
# print(John.df)

# print(John.statistic_in_one_col('Ток'))

# John.latex_table()
