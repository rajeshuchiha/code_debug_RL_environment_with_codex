def summarize_numbers(numbers):
    if not numbers:
        return {"total": 0, "average": 0, "min": None, "max": None}
    total = 0
    for value in numbers:
        total += value
    average = total / len(numbers)
    return {
        "total": total,
        "average": average,
        "min": min(numbers),
        "max": max(numbers),
    }
