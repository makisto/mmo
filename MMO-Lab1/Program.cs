using CsvHelper;
using Microsoft.Win32.SafeHandles;
using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;

namespace MMO_Lab1
{
    public class CsvLine
    {
        public int Param_1
        {
            get;
            set;
        }
        public int Param_2
        {
            get;
            set;
        }
        public int Class_number
        {
            get;
            set;
        }
    }

    class Program
    {
        static double EuclidDistance(CsvLine A, CsvLine B)
        {
            return Math.Sqrt(Math.Pow(A.Param_1 - B.Param_1, 2) + Math.Pow(A.Param_2 - B.Param_2, 2));
        }

        static double Weight(CsvLine U, CsvLine X)
        {
            return EuclidDistance(U, X);
        }

        static double Q_kvartic_function(double distance)
        {
            double r = Math.Min(distance / 10, 1);

            if (r > 1)
            {
                return 0;
            }
            else
            {
                return Math.Pow((1 - (r * r)), 2);
            }
        }

        static List<Tuple<double, CsvLine>> Get_neighbors(List<CsvLine> test_mas, CsvLine train_object, int k)
        {
            List<Tuple<double, CsvLine>> distances = new List<Tuple<double, CsvLine>>();
            List<Tuple<double, CsvLine>> neighbors = new List<Tuple<double, CsvLine>>();
            List<Tuple<double, CsvLine>> sorted_distances = new List<Tuple<double, CsvLine>>();

            foreach (CsvLine test in test_mas)
            {
                distances.Add(new Tuple<double, CsvLine>(Weight(train_object, test), test));
            }

            sorted_distances = distances.OrderBy(o => o.Item1).ToList();

            for (int i = 0; i < k; i++)
            {
                neighbors.Add(new Tuple<double, CsvLine>(Q_kvartic_function(sorted_distances[i].Item1), sorted_distances[i].Item2));
            }

            return neighbors;
        }
		private static void Main(string[] args)
        {
			if (args is null)
			{
				throw new ArgumentNullException(nameof(args));
			}

			Random rand = new Random();
            const double split = 1 / 3f;

            StreamReader reader = new StreamReader("data1.csv");
			CsvReader data_file = new CsvReader(reader, CultureInfo.InvariantCulture);
            data_file.Configuration.HasHeaderRecord = false;

            CsvLine[] csv_massive = data_file.GetRecords<CsvLine>().ToArray();

            List<int> classes = new List<int>();
            List<CsvLine> train_mas = new List<CsvLine>();
            List<CsvLine> test_mas = new List<CsvLine>();
            Dictionary<int, List<CsvLine>> full_csv_mas = new Dictionary<int, List<CsvLine>>();

			foreach (CsvLine csv_element in csv_massive)
            {
                if (!classes.Contains(csv_element.Class_number))
				{
                    classes.Add(csv_element.Class_number);
                }
            }

            foreach (int index in classes)
            {
                full_csv_mas.Add(index, new List<CsvLine>());
            }

            foreach (CsvLine csv_element in csv_massive)
            {
                full_csv_mas[csv_element.Class_number].Add(csv_element);
            }

            foreach (CsvLine csv_element in full_csv_mas.SelectMany(full_csv_mas_key => full_csv_mas_key.Value))
            {
                if (rand.NextDouble() < split)
				{
                    test_mas.Add(csv_element);
                }
                else
				{
                    train_mas.Add(csv_element);
                }
            }

            Console.WriteLine($"Test_mas count: {test_mas.Count}");
            Console.WriteLine($"Train_mas count: {train_mas.Count}");
            Console.WriteLine($"Total count: {test_mas.Count + train_mas.Count}");

            for (int k = 1; k < 101; k++)
            {
                int errors = 0;
                int total_objects = train_mas.Count;

                foreach (CsvLine train_object in train_mas)
                {
                    Dictionary<int, double> class_weight = new Dictionary<int, double>();
                    List<Tuple<double, CsvLine>> neighbors = Get_neighbors(test_mas, train_object, k);

                    foreach (int index in classes)
                    {
                        class_weight.Add(index, 0f);
                    }

                    foreach (Tuple<double, CsvLine> neighbor in neighbors)
                    {
                        class_weight[neighbor.Item2.Class_number] += neighbor.Item1;
                    }

                    int class_index = -1;
                    double max_weight = 0;
                    foreach (int index in classes)
                    {
                        if (max_weight < class_weight[index])
                        {
                            max_weight = class_weight[index];
                            class_index = index;
                        }
                    }

                    if (class_index == -1)
                    {
                        Dictionary<int, int> class_count = new Dictionary<int, int>();
                        foreach (int classIndex in classes)
                        {
                            class_count.Add(classIndex, 0);
                        }

                        foreach (Tuple<double, CsvLine> neighbor in neighbors)
                        {
                            class_count[neighbor.Item2.Class_number] += 1;
                        }

                        int count = 0;
                        foreach (int index in classes)
                        {
                            if (count < class_count[index])
                            {
                                count = class_count[index];
                                class_index = index;
                            }
                        }
                    }

                    if (train_object.Class_number != class_index)
                    {
                        errors++;
                    }
                }
                Console.WriteLine($"Correct precisions for {k} neighbors: {100 - errors * 100 / (double)total_objects:F2}%");
            }
        }
    }
}
