from cricket_tools.memory import get_memory, merge_with_memory, update_memory, clear_memory

clear_memory()
print("Start:", get_memory())

merged = merge_with_memory({"city": "Chennai"})
print("Merged1:", merged)

update_memory({"city": "Chennai", "season": "2021"})
print("After update:", get_memory())

merged2 = merge_with_memory({"metric": "runs", "n": 5})
print("Merged2:", merged2)