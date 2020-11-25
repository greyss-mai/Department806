----------------------------------------------------------------------------------
----------------------------------------------------------------------------------
library IEEE;
use IEEE.STD_LOGIC_1164.ALL;

entity add4 is
    Port ( a : in  STD_LOGIC_VECTOR (3 downto 0);
           b : in  STD_LOGIC_VECTOR (3 downto 0);
           sum : out  STD_LOGIC_VECTOR (3 downto 0);
           cout : out  STD_LOGIC);
end add4;

architecture STRUCTURE of add4 is

component full_adder
    Port ( a : in  STD_LOGIC;
           b : in  STD_LOGIC;
			  cin : in  STD_LOGIC;
           sum : out  STD_LOGIC;
           cout : out  STD_LOGIC);
end component;

signal c0, c1, c2, c3 : STD_LOGIC;

begin

c0 <= '0';
b_adder0: full_adder port map (a(0), b(0), c0, sum(0), c1);
b_adder1: full_adder port map (a(1), b(1), c1, sum(1), c2);
b_adder2: full_adder port map (a(2), b(2), c2, sum(2), c3);
b_adder3: full_adder port map (a(3), b(3), c3, sum(3), cout);

end STRUCTURE;

