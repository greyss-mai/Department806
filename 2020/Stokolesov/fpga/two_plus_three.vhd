library IEEE;


entity adder is
    Port ( a1 : in STD_LOGIC;
           a2 : in STD_LOGIC;
           b1 : in STD_LOGIC;
           b2 : in STD_LOGIC;
           sum1 : out STD_LOGIC;
           sum2 : out STD_LOGIC;
           sum_carry : out STD_LOGIC );
end adder;


architecture Behavioral of adder is
signal carry : STD_LOGIC;
begin
    sum1 <= a1 xor b1;
    carry <= (a1 and b1);
    sum2 <= a2 xor b2 xor carry;
    sum_carry <= (a2 and b2) or (a2 and carry) or (b2 and carry);
end Behavioral;