from cricket_tools.core import cricket_query
result = cricket_query("Rohit Sharma", role="batter")
print("\n=== Type of result ===")
print(type(result))

if isinstance(result, dict):
    print("\n=== Keys ===")
    print(result.keys())

    data = result.get("data")
    print("\n=== Type of result['data'] ===", type(data))
    if isinstance(data, dict):
        print("\n=== data keys ===", data.keys())
    elif hasattr(data, 'columns'):
        print("\n=== data columns ===", list(data.columns))
    elif isinstance(data, list) and len(data) > 0:
        print("\n=== first element keys ===", data[0].keys() if isinstance(data[0], dict) else type(data[0]))
else:
    print(result)
