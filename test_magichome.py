import magichome as mh


def test_checksum_matches_known_query():
    assert mh.cmd_state() == bytes([0x81, 0x8A, 0x8B, 0x96])


def test_rgb_command_is_9_byte_format():
    c = mh.cmd_rgb(80, 160, 200)
    assert len(c) == 9 and c[:8] == bytes([0x31, 80, 160, 200, 0, 0, 0xF0, 0x0F])
    assert c[8] == sum(c[:8]) & 0xFF


def test_parse_real_rgb_reply():
    st = mh.parse_state(bytes.fromhex("81 1e 23 61 30 32 ff 5a 19 00 0a 00 f0 f1"))
    assert st.power and st.mode == "rgb" and st.rgb == (255, 90, 25) and st.brightness_pct == 100


def test_parse_real_white_reply():
    st = mh.parse_state(bytes.fromhex("81 1e 23 61 30 32 00 00 00 45 0a 6a 0f 4d"))
    assert st.mode == "white" and (st.warm, st.cold) == (69, 106)
    assert st.brightness_pct == 69 and 4980 <= st.kelvin <= 5020


def test_white_split_respects_255_cap():
    for k in (2700, 4000, 5600, 6500):
        w, c = mh.kelvin_to_white(k, 100)
        assert w + c == 255
    assert mh.kelvin_to_white(2700, 100) == (255, 0)
    assert mh.kelvin_to_white(6500, 50) == (0, 128)


def test_kelvin_roundtrip():
    w, c = mh.kelvin_to_white(4600, 80)
    assert abs(mh.white_to_kelvin(w, c) - 4600) < 40
