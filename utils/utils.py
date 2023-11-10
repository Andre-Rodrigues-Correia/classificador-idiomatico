import random
import re
import requests
from bs4 import BeautifulSoup


class Utils:
    INDEX_SENTENCE = 1
    INDEX_LANGUAGE = 0

    @classmethod
    def get_test_dataset(cls):
        test_dataset = []
        with open("datasets/validate_corpus.txt", encoding="utf8") as read_file:
            for sentence in read_file:
                splited_sentences = sentence.split('\t')
                test_dataset.append((splited_sentences[cls.INDEX_SENTENCE], splited_sentences[cls.INDEX_LANGUAGE]))
        return test_dataset

    @classmethod
    def get_random_test_dataset(cls):
        num_lines_to_sample = 300000
        test_dataset = []

        with open("datasets/validate_corpus.txt", encoding="utf8") as read_file:
            lines = read_file.readlines()
            num_lines = len(lines)

            if num_lines_to_sample >= num_lines:
                return [(line.split('\t')[1], line.split('\t')[0]) for line in lines]

            sampled_indices = random.sample(range(num_lines), num_lines_to_sample)
            for index in sampled_indices:
                line = lines[index]
                splited_sentences = line.split('\t')
                test_dataset.append((splited_sentences[1], splited_sentences[0]))

        return test_dataset

    @classmethod
    def normalize_text(cls, text):
        normalized_text = re.sub(r'[^\w\s]', '', text)
        print(normalized_text)
        return normalized_text

    @classmethod
    def format_file_from_train(cls, language, directory):
        with open("datasets/formated_train_dataset.txt".format(language), "a+", encoding="utf-8") as write_file:
            with open(directory, encoding="utf8") as read_file:
                for sentence in read_file:
                    splited_sentences = sentence.split('\t')
                    normalized_text = cls.normalize_text(splited_sentences[cls.INDEX_SENTENCE])
                    write_file.write('{}\t{}'.format(language, normalized_text))

    @classmethod
    def get_portugal_gentiles(cls):
        url = "https://pt.wikipedia.org/wiki/Lista_de_gent%C3%ADlicos_de_Portugal"
        response = requests.get(url)
        soup = BeautifulSoup(response.content, "html.parser")

        gentilicos = []

        # Encontre a tabela com os gentílicos
        table = soup.find("table", {"class": "wikitable"})

        # Percorra as linhas da tabela
        for row in table.find_all("tr")[1:]:
            columns = row.find_all("td")
            if len(columns) >= 2:
                pais = columns[0].get_text(strip=True)
                gentilico = columns[1].get_text(strip=True)
                gentilicos.append((pais, gentilico))

        # Exibir os gentílicos obtidos
        with open("datasets/gentiles/portugal_gentiles.txt", "a+", encoding="utf-8") as write_file:
            for pais, gentilico in gentilicos:
                write_file.write('{}\n'.format(gentilico))

    @classmethod
    def get_brazil_gentiles(cls):
        url = "https://pt.wikipedia.org/wiki/Lista_de_gent%C3%ADlicos_do_Brasil"
        response = requests.get(url)
        soup = BeautifulSoup(response.content, "html.parser")

        brazil_gentilicos = []

        # Encontre a tabela com os gentílicos
        sections = soup.find_all("h2")

        for section in sections:
            # Extrai o nome do estado
            state_name = section.text.strip()

            # Encontra a lista de gentílicos dentro da seção
            ul = section.find_next("ul")

            # Itera sobre os itens da lista e extrai os gentílicos
            if ul:
                gentilicos = [li.text.strip() for li in ul.find_all("li")]
                print(f"Estado: {state_name}")
                print(f"Gentílicos: {', '.join(gentilicos)}\n")
                for gentilico in gentilicos:
                    brazil_gentilicos.append(gentilico)

        # Exibir os gentílicos obtidos
        with open("datasets/gentiles/brazil_gentiles.txt", "a+", encoding="utf-8") as write_file:
            for gentilico in brazil_gentilicos:
                try:
                    splited = gentilico.split('-')
                    text = re.sub(r'\s*\[.*?\]\s*', '', splited[1].strip())
                    if len(text) > 5:
                        write_file.write('{}\n'.format(text))
                except:
                    print('error  in gentilie {}'.format(gentilico))
