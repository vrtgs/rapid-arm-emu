use crate::helper::{
    call_compiled, compile, run_success, store_bool_as_x_reg, store_int_equals_as_x_reg, u64_const,
};
use emu_abi::processor_state::ProcessorState;
use exec_ir::{ExecIrBuilder, IConst, IntCmp, IntWidth};

mod helper;

#[test]
fn int_width_metadata_is_exact() {
    assert_eq!(IntWidth::from_bits(8), Some(IntWidth::W8));
    assert_eq!(IntWidth::from_bits(16), Some(IntWidth::W16));
    assert_eq!(IntWidth::from_bits(32), Some(IntWidth::W32));
    assert_eq!(IntWidth::from_bits(64), Some(IntWidth::W64));

    assert_eq!(IntWidth::from_bits(0), None);
    assert_eq!(IntWidth::from_bits(1), None);
    assert_eq!(IntWidth::from_bits(7), None);
    assert_eq!(IntWidth::from_bits(128), None);

    assert_eq!(IntWidth::W8.bits(), 8);
    assert_eq!(IntWidth::W16.bits(), 16);
    assert_eq!(IntWidth::W32.bits(), 32);
    assert_eq!(IntWidth::W64.bits(), 64);

    assert_eq!(IntWidth::W8.bytes(), 1);
    assert_eq!(IntWidth::W16.bytes(), 2);
    assert_eq!(IntWidth::W32.bytes(), 4);
    assert_eq!(IntWidth::W64.bytes(), 8);
}

#[test]
fn wrapping_add_sub_mul_and_neg_work() {
    let mut builder = ExecIrBuilder::default();

    let max = u64_const(&mut builder, u64::MAX);
    let one = u64_const(&mut builder, 1);
    let add_wrapped = builder.add(max, one);
    builder.store_x_reg::<0>(add_wrapped);

    let zero = u64_const(&mut builder, 0);
    let sub_wrapped = builder.sub(zero, one);
    builder.store_x_reg::<1>(sub_wrapped);

    let high_bit = u64_const(&mut builder, 0x8000_0000_0000_0000);
    let two = u64_const(&mut builder, 2);
    let mul_wrapped = builder.mul(high_bit, two);
    builder.store_x_reg::<2>(mul_wrapped);

    let five = u64_const(&mut builder, 5);
    let neg_five = builder.neg(five);
    builder.store_x_reg::<3>(neg_five);

    let mut state = ProcessorState::initial();
    run_success(builder, &mut state);

    assert_eq!(state.x_registers[0], 0);
    assert_eq!(state.x_registers[1], u64::MAX);
    assert_eq!(state.x_registers[2], 0);
    assert_eq!(state.x_registers[3], (!5_u64).wrapping_add(1));
}

#[test]
fn arithmetic_can_use_loaded_registers() {
    let mut builder = ExecIrBuilder::default();

    let x0 = builder.load_x_reg::<0>(IntWidth::W64);
    let x1 = builder.load_x_reg::<1>(IntWidth::W64);
    let sum = builder.add(x0, x1);
    builder.store_x_reg::<2>(sum);

    let x2 = builder.load_x_reg::<2>(IntWidth::W64);
    let three = u64_const(&mut builder, 3);
    let product = builder.mul(x2, three);
    builder.store_x_reg::<3>(product);

    let mut state = ProcessorState::initial();
    state.x_registers[0] = 9;
    state.x_registers[1] = 11;

    run_success(builder, &mut state);

    assert_eq!(state.x_registers[2], 20);
    assert_eq!(state.x_registers[3], 60);
}

#[test]
fn unsigned_division_handles_normal_and_zero_divisors() {
    let mut builder = ExecIrBuilder::default();

    let hundred = u64_const(&mut builder, 100);
    let seven = u64_const(&mut builder, 7);
    let quotient = builder.udiv(hundred, seven);
    builder.store_x_reg::<0>(quotient);

    let numerator = u64_const(&mut builder, 1234);
    let zero = u64_const(&mut builder, 0);
    let div_by_zero = builder.udiv(numerator, zero);
    builder.store_x_reg::<1>(div_by_zero);

    let mut state = ProcessorState::initial();
    run_success(builder, &mut state);

    assert_eq!(state.x_registers[0], 14);
    assert_eq!(state.x_registers[1], 0);
}

#[test]
fn signed_division_handles_normal_zero_and_overflow_cases() {
    let mut builder = ExecIrBuilder::default();

    let minus_seventeen = u64_const(&mut builder, (-17_i64).cast_unsigned());
    let five = u64_const(&mut builder, 5);
    let quotient = builder.sdiv(minus_seventeen, five);
    builder.store_x_reg::<0>(quotient);

    let numerator = u64_const(&mut builder, (-123_i64).cast_unsigned());
    let zero = u64_const(&mut builder, 0);
    let div_by_zero = builder.sdiv(numerator, zero);
    builder.store_x_reg::<1>(div_by_zero);

    let int_min = u64_const(&mut builder, i64::MIN.cast_unsigned());
    let minus_one = u64_const(&mut builder, (-1_i64).cast_unsigned());
    let overflow = builder.sdiv(int_min, minus_one);
    builder.store_x_reg::<2>(overflow);

    let mut state = ProcessorState::initial();
    run_success(builder, &mut state);

    assert_eq!(state.x_registers[0], (-3_i64).cast_unsigned());
    assert_eq!(state.x_registers[1], 0);
    assert_eq!(state.x_registers[2], i64::MIN.cast_unsigned());
}

#[test]
fn narrow_integer_ops_wrap_compare_and_divide_correctly() {
    let mut builder = ExecIrBuilder::default();

    let lhs = builder.iconst(IConst::u8(250));
    let rhs = builder.iconst(IConst::u8(10));
    let value = builder.add(lhs, rhs);
    store_int_equals_as_x_reg::<0>(&mut builder, value, IConst::u8(4));

    let lhs = builder.iconst(IConst::u16(0));
    let rhs = builder.iconst(IConst::u16(1));
    let value = builder.sub(lhs, rhs);
    store_int_equals_as_x_reg::<1>(&mut builder, value, IConst::u16(u16::MAX));

    let lhs = builder.iconst(IConst::u32(0x8000_0000));
    let rhs = builder.iconst(IConst::u32(2));
    let value = builder.mul(lhs, rhs);
    store_int_equals_as_x_reg::<2>(&mut builder, value, IConst::u32(0));

    let lhs = builder.iconst(IConst::u8(250));
    let rhs = builder.iconst(IConst::u8(10));
    let value = builder.udiv(lhs, rhs);
    store_int_equals_as_x_reg::<3>(&mut builder, value, IConst::u8(25));

    let lhs = builder.iconst(IConst::u8(250));
    let rhs = builder.iconst(IConst::u8(0));
    let value = builder.udiv(lhs, rhs);
    store_int_equals_as_x_reg::<4>(&mut builder, value, IConst::u8(0));

    let lhs = builder.iconst(IConst::i16(-9));
    let rhs = builder.iconst(IConst::i16(2));
    let value = builder.sdiv(lhs, rhs);
    store_int_equals_as_x_reg::<5>(&mut builder, value, IConst::i16(-4));

    let lhs = builder.iconst(IConst::i16(i16::MIN));
    let rhs = builder.iconst(IConst::i16(-1));
    let value = builder.sdiv(lhs, rhs);
    store_int_equals_as_x_reg::<6>(&mut builder, value, IConst::i16(i16::MIN));

    let value = builder.iconst(IConst::u8(1));
    let value = builder.neg(value);
    store_int_equals_as_x_reg::<7>(&mut builder, value, IConst::u8(255));

    let value = builder.iconst(IConst::u16(0x00ff));
    let value = builder.bitxor_imm(value, IConst::u16(0x0ff0));
    store_int_equals_as_x_reg::<8>(&mut builder, value, IConst::u16(0x0f0f));

    let mut state = ProcessorState::initial();
    run_success(builder, &mut state);

    for idx in 0..=8 {
        assert_eq!(state.x_registers[idx], 1, "x{idx}");
    }
}

#[test]
fn signed_and_unsigned_division_cover_more_rounding_edges() {
    let mut builder = ExecIrBuilder::default();

    let lhs = builder.iconst(IConst::i64(7));
    let rhs = builder.iconst(IConst::i64(-2));
    let value = builder.sdiv(lhs, rhs);
    builder.store_x_reg::<0>(value);

    let lhs = builder.iconst(IConst::i64(-7));
    let rhs = builder.iconst(IConst::i64(-2));
    let value = builder.sdiv(lhs, rhs);
    builder.store_x_reg::<1>(value);

    let lhs = builder.iconst(IConst::u64(u64::MAX));
    let rhs = builder.iconst(IConst::u64(u64::MAX));
    let value = builder.udiv(lhs, rhs);
    builder.store_x_reg::<2>(value);

    let lhs = builder.iconst(IConst::u64(7));
    let rhs = builder.iconst(IConst::u64(8));
    let value = builder.udiv(lhs, rhs);
    builder.store_x_reg::<3>(value);

    let mut state = ProcessorState::initial();
    run_success(builder, &mut state);

    assert_eq!(state.x_registers[0], (-3_i64).cast_unsigned());
    assert_eq!(state.x_registers[1], 3);
    assert_eq!(state.x_registers[2], 1);
    assert_eq!(state.x_registers[3], 0);
}

#[test]
fn integer_comparisons_cover_signed_unsigned_and_immediates() {
    let mut builder = ExecIrBuilder::default();

    let minus_one = builder.iconst(IConst::i64(-1));
    let plus_one = builder.iconst(IConst::u64(1));

    let cond = builder.icmp(IntCmp::Equal, minus_one, plus_one);
    store_bool_as_x_reg::<0>(&mut builder, cond);

    let cond = builder.icmp(IntCmp::NotEqual, minus_one, plus_one);
    store_bool_as_x_reg::<1>(&mut builder, cond);

    let cond = builder.icmp(IntCmp::SignedLessThan, minus_one, plus_one);
    store_bool_as_x_reg::<2>(&mut builder, cond);

    let cond = builder.icmp(IntCmp::SignedGreaterThanOrEqual, minus_one, plus_one);
    store_bool_as_x_reg::<3>(&mut builder, cond);

    let cond = builder.icmp(IntCmp::SignedGreaterThan, minus_one, plus_one);
    store_bool_as_x_reg::<4>(&mut builder, cond);

    let cond = builder.icmp(IntCmp::SignedLessThanOrEqual, minus_one, plus_one);
    store_bool_as_x_reg::<5>(&mut builder, cond);

    let cond = builder.icmp(IntCmp::UnsignedLessThan, minus_one, plus_one);
    store_bool_as_x_reg::<6>(&mut builder, cond);

    let cond = builder.icmp(IntCmp::UnsignedGreaterThanOrEqual, minus_one, plus_one);
    store_bool_as_x_reg::<7>(&mut builder, cond);

    let cond = builder.icmp(IntCmp::UnsignedGreaterThan, minus_one, plus_one);
    store_bool_as_x_reg::<8>(&mut builder, cond);

    let cond = builder.icmp(IntCmp::UnsignedLessThanOrEqual, minus_one, plus_one);
    store_bool_as_x_reg::<9>(&mut builder, cond);

    let cond = builder.icmp_imm(IntCmp::SignedLessThan, minus_one, IConst::u64(1));
    store_bool_as_x_reg::<10>(&mut builder, cond);

    let cond = builder.icmp_imm(IntCmp::UnsignedGreaterThan, minus_one, IConst::u64(1));
    store_bool_as_x_reg::<11>(&mut builder, cond);

    let cond = builder.icmp_imm(IntCmp::Equal, minus_one, IConst::i64(-1));
    store_bool_as_x_reg::<12>(&mut builder, cond);

    let mut state = ProcessorState::initial();
    run_success(builder, &mut state);

    assert_eq!(state.x_registers[0], 0);
    assert_eq!(state.x_registers[1], 1);
    assert_eq!(state.x_registers[2], 1);
    assert_eq!(state.x_registers[3], 0);
    assert_eq!(state.x_registers[4], 0);
    assert_eq!(state.x_registers[5], 1);
    assert_eq!(state.x_registers[6], 0);
    assert_eq!(state.x_registers[7], 1);
    assert_eq!(state.x_registers[8], 1);
    assert_eq!(state.x_registers[9], 0);
    assert_eq!(state.x_registers[10], 1);
    assert_eq!(state.x_registers[11], 1);
    assert_eq!(state.x_registers[12], 1);
}

#[test]
fn bitwise_integer_and_bool_ops_cover_reg_and_immediate_forms() {
    let mut builder = ExecIrBuilder::default();

    let a = builder.iconst(IConst::u64(0xca));
    let b = builder.iconst(IConst::u64(0xac));

    let value = builder.bitand(a, b);
    builder.store_x_reg::<0>(value);

    let value = builder.bitor(a, b);
    builder.store_x_reg::<1>(value);

    let value = builder.bitxor(a, b);
    builder.store_x_reg::<2>(value);

    let value = builder.bitand_imm(a, IConst::u64(0xf0));
    builder.store_x_reg::<3>(value);

    let value = builder.bitor_imm(b, IConst::u64(0x03));
    builder.store_x_reg::<4>(value);

    let value = builder.bitxor_imm(a, IConst::u64(0xff));
    builder.store_x_reg::<5>(value);

    let one = builder.iconst(IConst::u64(1));
    let zero = builder.iconst(IConst::u64(0));
    let true_bool = builder.icmp(IntCmp::NotEqual, one, zero);
    let false_bool = builder.icmp(IntCmp::Equal, one, zero);

    let value = builder.bitand(true_bool, false_bool);
    store_bool_as_x_reg::<6>(&mut builder, value);

    let value = builder.bitor(true_bool, false_bool);
    store_bool_as_x_reg::<7>(&mut builder, value);

    let value = builder.bitxor(true_bool, false_bool);
    store_bool_as_x_reg::<8>(&mut builder, value);

    let mut state = ProcessorState::initial();
    run_success(builder, &mut state);

    assert_eq!(state.x_registers[0], 0x88);
    assert_eq!(state.x_registers[1], 0xee);
    assert_eq!(state.x_registers[2], 0x66);
    assert_eq!(state.x_registers[3], 0xc0);
    assert_eq!(state.x_registers[4], 0xaf);
    assert_eq!(state.x_registers[5], 0x35);
    assert_eq!(state.x_registers[6], 0);
    assert_eq!(state.x_registers[7], 1);
    assert_eq!(state.x_registers[8], 1);
}

#[test]
fn select_handles_both_paths_and_bool_values() {
    let mut builder = ExecIrBuilder::default();

    let x0 = builder.load_x_reg::<0>(IntWidth::W64);
    let cond = builder.icmp_imm(IntCmp::NotEqual, x0, IConst::u64(0));

    let true_value = builder.iconst(IConst::u64(0xaaaa));
    let false_value = builder.iconst(IConst::u64(0xbbbb));
    let selected = builder.select(cond, true_value, false_value);
    builder.store_x_reg::<1>(selected);

    let one = builder.iconst(IConst::u64(1));
    let zero = builder.iconst(IConst::u64(0));
    let true_bool = builder.icmp(IntCmp::NotEqual, one, zero);
    let false_bool = builder.icmp(IntCmp::Equal, one, zero);
    let selected_bool = builder.select(cond, true_bool, false_bool);
    store_bool_as_x_reg::<2>(&mut builder, selected_bool);

    let compiled = compile(builder);

    let mut zero_state = ProcessorState::initial();
    zero_state.x_registers[0] = 0;
    assert_eq!(call_compiled(&compiled, &mut zero_state), 0);
    assert_eq!(zero_state.x_registers[1], 0xbbbb);
    assert_eq!(zero_state.x_registers[2], 0);

    let mut non_zero_state = ProcessorState::initial();
    non_zero_state.x_registers[0] = 1;
    assert_eq!(call_compiled(&compiled, &mut non_zero_state), 0);
    assert_eq!(non_zero_state.x_registers[1], 0xaaaa);
    assert_eq!(non_zero_state.x_registers[2], 1);
}
