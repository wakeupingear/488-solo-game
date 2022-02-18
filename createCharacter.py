import json
f = open("character.json", "r+")
data = json.load(f)

traits = ["Strength", "Dexterity", "Constitution", "Intelligence", "Wisdom", "Charisma"];

while True:
    name = input("Enter a name: ")
    if name == "":
        break
    fact = input("Enter a fact: ")
    if fact == "":
        continue
    data["name"] = name
    while True:
        prop = input("Enter a trait: ")
        if prop == "":
            break
        if prop in traits:
            data[prop] = input("Enter a value: ").lower()
        else:
            print("Invalid Trait")

    f.write(json.dumps(data))