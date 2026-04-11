def average_positive(numbers):
    positives = [number for number in numbers if number >= 0]
    return sum(positives) / len(numbers)
