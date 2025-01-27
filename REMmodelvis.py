import matplotlib.pyplot as plt
import csv

loss = list()
val_loss = list()

with open('names.csv', newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        loss.append(float(row['loss']))
        val_loss.append(float(row['val_loss']))

print(loss)
print(val_loss)

plt.figure(figsize=(10, 6))
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.ylim(0,0.5)
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True)

plt.savefig('plot.jpg', format='jpg')   