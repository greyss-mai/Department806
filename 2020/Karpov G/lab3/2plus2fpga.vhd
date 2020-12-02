library IEEE;
use IEEE.STD_LOGIC_1164.ALL;

-- Uncomment the following library declaration if using
-- arithmetic functions with Signed or Unsigned values
--use IEEE.NUMERIC_STD.ALL;

-- Uncomment the following library declaration if instantiating
-- any Xilinx primitives in this code.
--library UNISIM;
--use UNISIM.VComponents.all;

entity adder is
    Port ( a1 : in  STD_LOGIC;
           a2 : in  STD_LOGIC;
           b1 : in  STD_LOGIC;
           b2 : in  STD_LOGIC;
           sum1 : out  STD_LOGIC;
           sum2 : out  STD_LOGIC;
           carry : out  STD_LOGIC);
			  
end adder;

architecture Behavioral of adder is
signal buffCarry : STD_LOGIC;
begin
sum1 <= a1 xor b1;
buffCarry <= (a1 and b1);
sum2 <= a2 xor b2 xor buffCarry;
carry <= ((a2 xor b2) and buffCarry) or (a2 and b2);

end Behavioral;

