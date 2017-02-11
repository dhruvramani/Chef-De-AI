import re
import requests
from bs4 import BeautifulSoup

def main():
    text = ""
    ingredientsStr = ""
    instructionStr = ""
    with open("all.txt", 'r') as file :
        links = file.read().split('\n')
        for i in links:
            print("Crawling {}".format(i))
            req = requests.get(i)
            soup = BeautifulSoup(req.text, "html.parser")
            ingredients = soup.select('.ingredient')
            instructions = soup.select('.instruction')
            cleanr = re.compile('<.*?>')

            for j in ingredients :
                string = re.sub(cleanr, '', str(j.contents[0]))
                text += string + "\n"
                ingredientsStr += string + "\n"

            for j in instructions :
                string = re.sub(cleanr, '', str(j.contents[0]))
                text += string + "\n"
                instructionStr += string + "\n"
            text += "\n\n"
            ingredientsStr += "\n\n"
            instructionStr += "\n\n"
    
    with open("all_text.txt", 'w') as main :
        main.write(text)
    with open("all_ingredients.txt", 'w') as ingredients :
        ingredients.write(ingredientsStr)
    with open("all_instructions.txt", 'w') as instruction :
        instruction.write(instructionStr)

if __name__ == '__main__':
    main()