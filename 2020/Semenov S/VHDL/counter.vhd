----------------------------------------------------------------------------------
-- Company: 
-- Engineer: 
-- 
-- Create Date:    20:05:43 03/15/2020 
-- Design Name: 
-- Module Name:    counter - Behavioral 
-- Project Name: 
-- Target Devices: 
-- Tool versions: 
-- Description: 
--
-- Dependencies: 
--
-- Revision: 
-- Revision 0.01 - File Created
-- Additional Comments: 
--
----------------------------------------------------------------------------------
library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
use IEEE.STD_LOGIC_ARITH.ALL;
use IEEE.STD_LOGIC_UNSIGNED.ALL;

-- Uncomment the following library declaration if using
-- arithmetic functions with Signed or Unsigned values
--use IEEE.NUMERIC_STD.ALL;

-- Uncomment the following library declaration if instantiating
-- any Xilinx primitives in this code.
--library UNISIM;
--use UNISIM.VComponents.all;

entity counter is
    Port ( CLOCK : in  STD_LOGIC;
           DIRECTION : in  STD_LOGIC;
           COUNT_OUT : out  STD_LOGIC_VECTOR (3 downto 0));
end counter;

architecture Behavioral of counter is
	signal count_int : std_logic_vector(3 downto 0) := "0000";
	signal CLK_1Hz: std_logic_vector(23 downto 0);

begin
process (CLOCK)
begin
       if CLOCK='1' and CLOCK'event then
		 if CLK_1Hz < "110011011111111001100000" then
					CLK_1Hz <= CLK_1Hz + 1;
				else 
					CLK_1Hz <= (others => '0');
					if DIRECTION='1'  then
						count_int <= count_int + 1;
					else
						count_int <= count_int - 1;
					end if;
					
			end if;
           
       end if;
end process;
COUNT_OUT <=count_int;
end Behavioral;

