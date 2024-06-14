from ..tokenization import SpaceGroupEncoder

def test_space_group_encoder():
    all_group_encoder = SpaceGroupEncoder.from_sg_set(range(1, 231))
    all_groups = set()
    for group_number in range(1, 231):
        group = tuple(all_group_encoder[group_number])
        if group in all_groups:
            raise ValueError(f"Duplicate group: {group}")
        all_groups.add(group)
    print("All space groups are unique.")

if __name__ == "__main__":
    test_space_group_encoder()